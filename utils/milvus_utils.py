from pymilvus import MilvusClient

from config.milvus_config import MilvusConfig, milvus_config

_milvus_client = None
def get_milvus_client():
    """
    获取全局单例的MilvusClient对象
    :return: MilvusClient实例
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数
    :param value: 转义前的原始字符串
    :return: str:转义后的安全字符串
    """
    # 转义反斜杠（\ -> \\) 和双引号（" -> \”） 单引号(' -> \')
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


if __name__ == '__main__':
    collections = get_milvus_client()
    print(f"📚 当前存在的集合列表：{collections}")
