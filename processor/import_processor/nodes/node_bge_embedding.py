import json
import logging
from typing import List, Dict
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        #1.参数处理
        chunks = self._step_1_validate_paths(state)

        #2.数据向量化
        output_data = self._step_2_bge_embedding(chunks)
        print(output_data)
        #3.返回结果
        state["chunks"] = output_data
        return state

    def _step_1_validate_paths(self, state : ImportGraphState)-> List[Dict]:
        print("node_bge_embedding: 参数校验")
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks
    def _step_2_bge_embedding(self, chunks: List[Dict]) -> List[Dict]:
        """
        将item_name和content转化为向量数据（稀疏和稠密）
        """
        print("node_bge_embedding: 数据向量化")
        output_data = []
        batch_size= 5 #批量处理
        for i in range(0,len(chunks),batch_size):
            five_ready_xlh_texts = []
            batch_texts = chunks[i:i+batch_size] #第一次从0取到4，共5块
            for doc in batch_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                five_ready_xlh_texts.append(f"{item_name}\n{content}" if item_name else content)

            embeddings = generate_embeddings(five_ready_xlh_texts) #向量化结果

            for j,doc in enumerate(batch_texts):
                item = doc.copy()
                dense = embeddings["dense"][j]
                item["dense_vector"] = dense
                sparse = embeddings["sparse"][j]
                item["sparse_vector"] = sparse
                output_data.append(item)

        print(output_data)


if __name__ == '__main__':
    node = NodeBGEEmbedding()
    with open("F:\output\hak180产品安全手册\hak180产品安全手册_chunks.json","r",encoding="utf-8") as f:
        chunks_content = f.read()

    json_state = json.loads(chunks_content)
    init_state = {
        "chunks": json_state
    }
    response = node(init_state)

    dumps = json.dumps(response,ensure_ascii=False,indent=4)
    print(dumps)
