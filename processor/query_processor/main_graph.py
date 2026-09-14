from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from tool.logger import logger
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyd import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState

load_dotenv()


class KBQueryWorkflow:
    def __init__(self):
        #1.工作流状态
        self.workflow =  StateGraph(QueryGraphState)

        #2.实例化所有节点
        self._init_nodes()

        #3.注册所有节点
        self._register_nodes()

        #4.设置路由规则
        self._setup_route()

        #5.编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    #实例化节点
    def _init_nodes(self):
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    #注册节点
    def _register_nodes(self):
        # 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
        # lambda x:x 含义：接收 state 并原样返回，是最轻便的 “无逻辑传递” 方式；
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)  # 确认主体
        self.workflow.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点（状态不变）
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)  # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)  # 联网搜索
        self.workflow.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("node_rrf", self.node_rrf)  # 排序
        self.workflow.add_node("node_rerank", self.node_rerank)  # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output)  # 生成

    #主体名称确定后的条件路由函数
    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        if state.get("answer"):
            return "node_answer_output"

        #否则继续搜索流程
        return ["node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"]

    #设置路由规则
    def _setup_route(self):
        #1.设计入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        #2.注册条件路由边界
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
   {
                "node_answer_output": "node_answer_output",
                "node_search_embedding":"node_search_embedding",
                "node_search_embedding_hyde":"node_search_embedding_hyde",
                "node_web_search_mcp": "node_web_search_mcp",
            }
        )
        #3.多路搜索结果合并
        self.workflow.add_edge("node_search_embedding","node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde","node_rrf")
        self.workflow.add_edge("node_web_search_mcp","node_rrf")

        #4.排序 -> 重排 —> 生成 -> 结束
        self.workflow.add_edge("node_rrf","node_rerank")
        self.workflow.add_edge("node_rerank","node_answer_output")
        self.workflow.add_edge("node_answer_output",END)

    #编译工作流
    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    #统一执行入口，可切换是否流式输出（invoke/stream）
    def run(self,initial_state: QueryGraphState,stream :bool = False) -> QueryGraphState:
        if not self._compiled_app:
            self.compile()

        self._compiled_app.get_graph().print_ascii()

        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

#测试用例
if __name__ == "__main__":

    #定义初始状态
    init_state = {
        "original_query":"烫金机如何使用？",
        "session_id" :123123
    }

    workflow = KBQueryWorkflow()
    #写完再回答
    # final_state = workflow.run(init_state)
    # logger.info(final_state)
    #流式输出（边写边答）
    for chunk in workflow.run(init_state,stream=True):
        logger.info(chunk)
        print(chunk)
    #打印编译后的图结构
    logger.info(workflow.compile().get_graph().print_ascii())








