from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 1.参数处理

        # 2.标题切（初切）

        # 3.内容切（
        return state