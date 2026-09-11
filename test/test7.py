from utils.embedding_utils import generate_embeddings

#调试1、直接调用bge_m3客户端调用向量化方法，返回带有稠密和稀疏向量的列表
embeddings = generate_embeddings(["你好","hello"])
print(embeddings)
print("\n")
#测试2、用封装的generrate_embeddings方法，返回带有稠密和稀疏向量的具体浮点列表
embeddings = generate_embeddings(["你好","hello"])
print(embeddings["dense"][0])
print(embeddings["sparse"][0])