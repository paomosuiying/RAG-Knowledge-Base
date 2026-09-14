from pymongo import MongoClient


mongo_client = MongoClient("mongodb://192.168.100.100:27017")
db = mongo_client["test"]
#创建集合
# db.create_collection("classes")

# 插入数据
db["classes"].insert_one({"name":"wanwu","age": 12})

#查询数据
find_result = db["classes"].find()
print(find_result)
for doc in find_result:
    print(doc)
    print(doc["name"])
