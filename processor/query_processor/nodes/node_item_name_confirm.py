import json
from typing import Dict, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import CustomJSONEncoder
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        logger.info(f"【{self.name}】节点逻辑")

        #1.参数校验
        session_id,original_query = self._step_1_validate_parm(state)  #会话id和本次问题

        #2.获取历史对话（记忆）
        history = get_recent_messages(session_id)
        state["history"] = history

        #3.保存用户信息
        message_id = save_chat_message(session_id, "user",original_query )

        #4.模型提取主体:rewritten_query(改写的问题),item_name(模型识别到的设备主体）
        extract_res = self._step_4_extract_info(original_query,history)
        item_names = extract_res["item_names"] #模型可识别的设备主体名称
        rewritting_query = extract_res["rewritten_query"]#模型改写后的问题
        state["rewritting_query"] = rewritting_query
        state["item_names"] = item_names

        #5.向量搜索（搜索知识库）和6.搜索结果对齐（整理）
        align_result = {}
        if len(item_names) >0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._Step_6_align_item_names(query_results)
        else:
            logger.info("Node:未能提取到商品名，跳过向量检索")
        #7.状态state信息整理
        state = self._step_7_chunk_confirmation(state,align_result,history,message_id)

        #8.写入历史会话（记忆）
        self._step_8_write_history(state,session_id,rewritting_query,message_id)

        return state

    def _step_1_validate_parm(self, state):
        print("node_item_name_confirm:步骤1 参数校验")
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query :
            raise ValueError("核心参数original_query缺失")
        return session_id,original_query

    def _step_4_extract_info(self, original_query, history) -> Dict:
        print("node_item_name_confirm:步骤4 模型提取主体")
        try:
            #llm客户端
            ai_client = get_llm_client()

            # 2. 构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            # 3. 处理和动态拼接提示词
            # 为了把大括号当作 “普通字符” 保留下来，用双大括号 {{ 表示普通的左大括号 {，双大括号 }} 表示普通的右大括号 }。
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=original_query
            )

            # 4. 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 5. 调用LLM客户端，发起请求获取结果
            response = ai_client.invoke(messages)
            content = response.content

            # 6. 数据清洗：处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            # 7. 数据解析：将JSON字符串转为字典
            result = json.loads(content)

            # 8. 健壮性处理：确保字段存在
            # 确保返回结果包含item_names字段，无则设为空列表
            if "item_names" not in result:
                result["item_names"] = []
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
            if "rewritten_query" not in result:
                result["rewritten_query"] = original_query

            # 9. 给item_names 去除空格
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

            # 10、返回解析后的提取结果
            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": original_query}

    def _step_5_vectorize_and_query(self, item_names ) -> List[Dict]:
        print("node_item_name_confirm:步骤5 向量检索,检索出来对应item_name的向量数据库中的相似的商品名的列表")
        result : List[Dict] = []

        #milvus的客户端，集合名称
        milvus_client = get_milvus_client()
        if not milvus_client:
            logger.error("连接 Milvus 失败")
            return result
        collection_name = milvus_config.item_name_collection

        #条件向量化
        embeddings = generate_embeddings(item_names)
        try:
            #相似性匹配
            for i in range(len(item_names)):
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]
                reqs = create_hybrid_search_requests(dense_vector = dense_vector, sparse_vector = sparse_vector ,limit= 5)
                search_res = hybrid_search(
                    client=milvus_client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8,0.2),
                    norm_score = True,
                    output_fields=["item_name"]
                )
                #结果处理
                matches = []
                if search_res and len(search_res)>0:
                    # print("此处是视图识别的向量搜索结果:{}".format(search_res))
                    for hit in search_res[0]:
                        #将searc_res[0]中的hit对象转换为字典
                        matches.append({
                            "item_name": hit.entity.get("item_name"),
                            "score": hit.get("distance")
                        })
                result.append({
                    "item_name": item_names[i],
                    "matches": matches #相似性匹配结果
                })
        except Exception as e:
            logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")
        #返回结果
        return result


    def _Step_6_align_item_names(self, query_results:List[Dict])-> Dict:
        print("node_item_name_confirm:步骤6 搜索结果对齐")
        logger.info(f"步骤6：获得待处理的数据源：{query_results}")
        # 1、初始化确认商品名列表（符合高置信度规则的商品名）
        confirmed_item_names: List[str] = []
        # 2、初始化候选商品名列表（低置信度，需用户确认的商品名）
        options: List[str] = []

        for res in query_results:
            # 提取原始的数据，商品名和匹配结果
            extracted_name = (res.get("extracted_name", "") or "").strip()
            # 获取匹配的商品名，无就获取空列表
            matches = res.get("matches", []) or []
            # 若无匹配结果，直接跳过当前商品名的对齐
            if not matches:
                continue

            #规则a: 只有一个高置信度结果（>0.8）→ 直接确认该商品名
            #规则b: 如果多条匹配结果评分超过0.8 → 优先取与原始提取名相同的，无则取分数最高的
            # 筛选高置信度匹配结果：>0.8
            high = [m for m in matches if m.get("score", 0) > 0.8]
            # 筛选中置信度匹配结果：>=0.6
            mid = [m for m in matches if m.get("score", 0) >= 0.6]
            if len(high) > 0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                    continue
            #规则c: 如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
            if len(mid) > 0:
                # 取中置信度结果的前5个，加入候选列表
                for m in mid[:3]:
                    options.append(m.get("item_name"))
        #规则d: 如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
        # 返回最终对齐结果：确认列表和候选列表均做去重处理（list(set())）
        print(confirmed_item_names,options)
        return {
            "confirmed_item_names":[],#确认后的商品名称(>0.8)
            "options":[],#可能低分的商品民名称（<0.6)
        }

    def _step_7_chunk_confirmation(self, state, align_result :Dict, history, message_id):
        print("node_item_name_confirm:步骤7 状态state信息整理")#根据第六步的高分低分结果对齐整理
        confirmed = align_result.get("confirmed_item_names")
        options = align_result.get("options")
        #1.有命中(>0.8)
        if confirmed:
            #更新会话信息，将命中结果更新到与本次命中结果有关的所有的之前的会话中（session_id,_id）
            ids_to_update =[]
            for msg in history:
                if not msg.get("item_names"):
                    mid = msg.get("_id")
                    if mid :
                        ids_to_update.append(str(mid))
            if ids_to_update:
                update_message_item_names(ids_to_update,confirmed)

            #封装结果
            state["item_names"] =confirmed
            state["answer"] = ""

        #2.有备选(>=0.6)
        if options:
            state["item_names"] =[]
            options_str = "、".join(options)
            state["answer"] = f"您是想问以下哪个产品{options_str}？请明确下型号。"

        #3.无命中(<0.6)
        #封装结果
        if not confirmed and not options:
            state["answer"] = "抱歉，未能找到相关的产品"
            state["item_names"] = []  #相似度高于0.8才封装state进入后续节点

        #4.返回state
        return state


    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("node_item_name_confirm:步骤8 写入历史会话(更新）")

        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回会话状态（下一步使用）
        return state


if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "original_query": "烫金机怎么用？",
        "session_id": "1234567890"

    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4,cls=CustomJSONEncoder)
    # 输出
    logger.info(json_state)