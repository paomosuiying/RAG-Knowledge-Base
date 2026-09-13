import json
import logging
from typing import Dict, Any, List
from pymilvus import DataType
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string

class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        #1.数据校验
        chunks_json_data,vector_dimension = self._step_1_check_inputs(state)

        #2.结构准备
        milvus_client = self._step_2_prepare_collectiom(state,vector_dimension)

        #3.清理可能的冗余数据（幂等性）
        self._step_3_clean_old_data(milvus_client,chunks_json_data)
        #4.数据入库，返回数据库主键
        update_chunks = self._step_4_insert_data(milvus_client,chunks_json_data)
        #5.更新状态
        state["chunks"] = update_chunks

        return state

    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数： state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：tuple - (校验通过的切片列表, 稠密向量维度)
        异常：任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理
        """
        print("node_import_milvus:步骤1数据校验")
        # 校验1：chunks非空
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 提取向量维度
        vector_dimension = len(first_chunk['dense_vector'])
        return chunks, vector_dimension


        pass

    def _step_2_prepare_collectiom(self, state, vector_dimension):
        print("node_import_milvus:步骤2结构准备")
        """
                步骤2：Milvus客户端连接+集合准备
                核心逻辑：
                    1. 获取Milvus单例客户端，验证连接有效性
                    2. 集合不存在则自动创建（Schema+索引），存在则直接复用
                参数：vector_dimension: int - 稠密向量维度（步骤1提取）
                返回：MilvusClient - 已连接、集合准备完成的客户端实例
                异常：客户端获取失败/集合名称未配置，抛出异常终止流程
        """

        # 1. 获取milvus客户端对象
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 2. 集合不存在则创建
        collections_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collections_name):
            self._create_chunks_collection(collections_name, milvus_client, vector_dimension)

        return milvus_client

    def _create_chunks_collection(self, collections_name, milvus_client, vector_dimension):

        # 1. 创建schem
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 2. 创建列
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 3. 创建索引
        index_params = milvus_client.prepare_index_params()
        # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度（语义检索常用）
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积（IP），适配稀疏向量检索
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )


    def _step_3_clean_old_data(self, client, chunks_json_data):
        """
                幂等清理
                基于每个片段的file_title进行旧数据的清理
                :param client: milvus客户端
                :param chunks_json_data: chunks数据
                :return:
                """
        print("node_import_milvus:步骤3清理冗余数据")
        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self._clear_chunks_by_file_title(client, file_title)

        def _clear_chunks_by_file_title(self, client, file_title):

            try:
                file_title = escape_milvus_string(file_title)
                client.delete(
                    collection_name=milvus_config.chunks_collection,
                    filter=f"file_title=='{file_title}'")
            except Exception as e:
                self.logger.error(f"Milvus 数据删除失败: {str(e)}")
                raise MilvusError(f"Milvus 数据删除失败: {str(e)}")


    def _clear_chunks_by_file_title(self, client, file_title):
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("node_import_milvus:步骤4数据入库")
        """步骤4：批量插入切片数据到Milvus+主键回填
                核心逻辑：
                    1. 批量插入数据：提升入库效率，减少Milvus连接次数
                    2. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
                参数：client - MilvusClient实例
                         chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
                返回：List[Dict[str, Any]] - 回填了chunk_id的切片列表
        """
        # 1. 预处理数据：移除手动chunk_id，避免与Milvus自增主键冲突
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()

            # 补充 part 字段
            if "part" not in item_copy:
                item_copy["part"] = 0

            # 添加到待插入列表
            data_to_insert.append(item_copy)

        # 2. 执行批量插入
        insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get('insert_count', 0)

        # 3. 主键回填：将Milvus生成的chunk_id回填到原始切片
        inserted_ids = insert_result.get('ids', [])
        if inserted_ids:
            for idx, item in enumerate(chunks_json_data):
                item['chunk_id'] = str(inserted_ids[idx])

        return chunks_json_data

if __name__ == '__main__':
    setup_logging()

    json_path = r"F:\output\hak180产品安全手册\hak180产品安全手册_new_new_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))