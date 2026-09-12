import json
from os import system
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pyarrow import system_memory_pool

from config.lm_config import lm_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from processor.query_processor.prompt.item_name_recognition import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from utils.embedding_utils import generate_embeddings


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
        item_name = self._step_3_call_llm(file_title, chunks)

        # 4.回填数据（item_name -> chunks)
        self._step_4_update_chunks(state, item_name, chunks)

        # 5.主体名称向量化
        dense_vector, sparse_vector = self._step_5_generate_embedding(item_name)

        # 6.存入milvus向量库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    def _step_1_get_inputs(self, state) -> (str, List[Dict]):
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

    def _step_3_call_llm(self, file_title, context, user_prompt=None)->str:
        print("node_item_name_recognition: 步骤3 模型识别")
        if not context :
            return file_title
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
            HumanMessage(content = ITEM_NAME_USER_PROMPT_TEMPLATE)
        ]

        #调用llm
        response = llm_ai.invoke(messages)

        #解析:数据清洗
        item_name=getattr(response, "choices","").strip()
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        #兜底
        if not item_name:
            item_name = file_title

        return item_name

    def _step_4_update_chunks(self, state, item_name, chunks):
        print("node_item_name_recognition: 步骤4 回填数据")
        state["item_name"] = item_name

        for chunk in chunks:
            chunk["item_name"] = item_name

        return state

    def _step_5_generate_embedding(self, item_name)->(List, List):
        print("node_item_name_recognition: 步骤5 主体名称向量化,返回稠密和稀疏数据")

        embeddings = generate_embeddings([item_name])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense, sparse

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        print("node_item_name_recognition: 步骤6 存入milvus向量库")

        return state


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
    for chunk in process["chunks"]:
        print(f'{chunk.get("title")} {chunk.get("item_name")}')
    print(process)