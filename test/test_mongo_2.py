from utils.mongo_history_utils import get_history_mongo_tool

#通过工具获取mongo的客户端
mongo_client = get_history_mongo_tool()

#向chat_message表插入数据
mongo_client.chat_message.insert_one({"session": "123", "message": "hello","ts": 1234567890})