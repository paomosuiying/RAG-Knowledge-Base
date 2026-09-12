import json
from os import system
from typing import List, Dict, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from pymilvus import DataType
from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from processor.query_processor.prompt.item_name_recognition import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        # 1.参数处理
        file_title, chunks = self._step_1_get_inputs(state)

        # 2.上下文拼接
        context = self._step_2_build_context(file_title, chunks)

        # 3.模型识别（总结）
        item_name = self._step_3_call_llm(file_title, context)

        # 4.回填数据（item_name -> chunks)
        self._step_4_update_chunks(state, chunks, item_name)

        # 5.主体名称向量化
        dense_vector, sparse_vector = self._step_5_generate_embedding(item_name)

        # 6.存入milvus向量库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    def _step_1_get_inputs(self, state:ImportGraphState) -> Tuple[str, List[Dict]]:
        print("node_item_name_recognition: 步骤1 参数处理")
        file_title = state["file_title"]
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)
        chunks = state["chunks"]
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        return file_title, chunks

    def _step_2_build_context(self, file_title, chunks : List[Dict])->str:
        print("node_item_name_recognition: 步骤2 上下文拼接")
        #上下文限制的片段
        k = self.config.item_name_chunk_k
        #上下文限制的长度
        chunk_size = self.config.item_name_chunk_size

        parts : List[str] = []
        total_chars = 0
        for index , chunk in enumerate(chunks[:k],start=1):
            chunk_title = chunk.get("title","").strip()
            chunk_content = chunk.get("content","").strip()

            #格式化
            piece = f"【切片:{index}】\n标题:{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            #计算长度
            total_chars += len(piece)

            #检测长度
            if total_chars > chunk_size:
                break

        #阶段处理
        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]
        return final_context

    def _step_3_call_llm(self, file_title:str, context:str)->str:
        print("node_item_name_recognition: 步骤3 模型识别")
        if not context :
            return file_title
        try:
            #llm
            llm_ai = ChatOpenAI(
                model=lm_config.llm_model,
                api_key = lm_config.api_key,
                base_url = lm_config.base_url,
                temperature= lm_config.llm_temperature,
                extra_body={"enable_thinking" : False}
            )
            # #提示词
            # prompt = f"""
            # 请从以下信息中识别出商品名称与型号：
            # 文件名：{file_title}
            # 正文切片（用于辅助识别）：
            # {context}
            # 要求：
            # 1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
            # 2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
            # 3. 如果无法识别商品名称,请返回空字符串。
            # """
            # message = [
            #     SystemMessage("你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。")
            #     HumanMessage(prompt)
            # ]
            #创建消息对象(系统提示词、用户提示词)
            messages = [
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(
                    content=ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                        file_title=file_title,
                        context=context
                    )
                )
            ]

            #调用llm
            response = llm_ai.invoke(messages)

            #解析:数据清洗
            item_name = (response.content or "").strip()
            item_name = (item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", ""))
            #兜底
            if not item_name:
                item_name = file_title

            return item_name
        except Exception  as e:
            self.logger.error(f"大模型调用异常：{e}")
            return file_title
    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        print("node_item_name_recognition: 步骤4 回填数据")
        # 1. 遍历所有切片，为每个切片添加商品名字段
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 2. 同步更新state中的切片列表
        state["chunks"] = chunks

        # 3. 将商品名称存入全局状态
        state["item_name"] = item_name

    def _step_5_generate_embedding(self, item_name:str)->(List, List):
        print("node_item_name_recognition: 步骤5 主体名称向量化,返回稠密和稀疏数据")

        embeddings = generate_embeddings([item_name])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense, sparse

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector,sparse_vector):
        print("node_item_name_recognition: 步骤6 存入milvus向量库")
        """
        state:流程状态对象，用于最终状态同步
        file_title:处理后的标题
        item_name: 识别后的商品名称
        dense_vector:步骤五的稠密向量
        sparse_vector:步骤五的稀疏向量
        """
        try:
            #1.客户端获取
            milvus_client = get_milvus_client()
            if not milvus_client:
                self.logger.warning("无法获取 Milvus 客户端（连接失败），跳过数据保存。")
                return

            #2.集合初始化
            collection_name = milvus_config.item_name_collection
            if not milvus_client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, milvus_client)

            # 3.幂等性处理(删除同名表数据)
            #3.1转义商品名称
            safe_item_name =escape_milvus_string(item_name)
            #3.2构建过滤表达式：item_name等于目标值
            filter_expr = f'item_name=="{safe_item_name}"'
            #3.3删除符合条件的数据
            milvus_client.delete(collection_name = collection_name, filter = filter_expr )

            # 4.准备插入的数据
            data = {
                "file_title" : file_title,#文件标题
                "item_name" : item_name,#商品名称
                # "dense_vector" : dense_vector,#稠密向量
                # "sparse_vector" : sparse_vector#稀疏向量
            }

            #稠密向量非空则添加
            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            #稀疏向量非空则添加
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            # 5.数据插入
            milvus_client.insert(collection_name = collection_name, data = data)
            milvus_client.load_collection(collection_name = collection_name)#将数据从存储引擎加入到检索引擎
            #6.把商品名放入state
            state["item_name"] = item_name
        except Exception as e:
            self.logger.warning(f"数据存入Milvus失败: {str(e)}",exc_info = True)

    #步骤6 方法1：创建数据定义
    def _create_item_name_collection(self, collection_name, milvus_client):
        # 创建Schema（数据结构定义）
        # auto_id=True：主键自动生成；enable_dynamic_field=True：支持动态字段（允许插入 Schema 中未定义的字段）
        schema = milvus_client.create_schema(auto_id = True,enable_dynamic_field = True)
        # 添加主键字段（INT64类型，自增）
        schema.add_field(
            field_name = "pk",
            datatype = DataType.INT64,
            is_primary = True,
            auto_id = True
        )
        # 添加文件标题字段（VARCHAR类型，最大长度65535）
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        # 添加商品名称字段（VARCHAR类型，最大长度65535）
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        # 添加稠密向量字段（FLOAT_VECTOR类型，1024维，BGE-M3模型固定维度）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        # 添加稀疏向量字段（SPARSE_FLOAT_VECTOR类型，变长，适配BGE-M3的稀疏向量）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3. 构建索引参数（提升向量检索性能）
        index_params = milvus_client.prepare_index_params()
        # 为稠密向量创建索引（IVF_FLAT：兼容性好，适合小数据量）
        # 核心是 “先聚类分桶、再桶内暴力精确检索”。
        index_params.add_index(
            field_name="dense_vector",  # 字段名
            index_name="dense_vector_index",  # 索引名
            index_type="IVF_FLAT",  # 索引类型（分组+精准收索）
            metric_type="COSINE",  # 相似度计算方式（余弦相似度）
            params={"nlist": 128}  # 聚类数（影响检索精度/速度）
        )

        # 为稀疏向量创建索引（SPARSE_INVERTED_INDEX：稀疏向量专用索引）
        index_params.add_index(
            field_name="sparse_vector",  # 字段名
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type="IP",  # 相似度计算方式（内积）
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # 高效的稀疏检索算法

                "normalize": True,
                # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度

                "quantization": "none"
                # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了
                # "quantization": "none" → 存储原始向量，不压缩
                # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
            })

        # 4. 创建集合（Schema + 索引）
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

if __name__ == '__main__':
    node = NodeItemNameRecognition()

    path = "F:\output\hak180产品安全手册\hak180产品安全手册_chunks.json"

    with open(path, 'r' , encoding="utf-8") as f :
        chucks_json_data = f.read()

    chunks =json.loads(chucks_json_data)
    init_state = {
        "file_title" : "hak180产品安全手册",
        "chunks" : chunks
    }

    process = node.process(init_state)
    # for chunk in process["chunks"]:
    #     print(f'{chunk.get("title")} {chunk.get("item_name")}')
    # print(process)