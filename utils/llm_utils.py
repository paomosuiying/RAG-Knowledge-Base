from langchain_openai import ChatOpenAI
from config.lm_config import lm_config

_llm_client_cache ={}

def get_llm_client(model:str|None=None , json_model:bool=False):
    m = model or lm_config.llm_model

    key = (m , json_model)

    if key  in _llm_client_cache:
        return _llm_client_cache[key]

    model_kwargs = {}
    if json_model :
        model_kwargs["response_format"] ={"type" : "json_object" }

    #返回模式
    client = ChatOpenAI(
        model = m,
        temperature = lm_config.llm_temperature,
        base_url = lm_config.base_url,
        api_key = lm_config.api_key,
        model_kwargs=model_kwargs
    )

    _llm_client_cache[key] = client
    return client


#测试模型连接
if __name__ == "__main__":
    client = get_llm_client()
    result = client.invoke("你好,你是谁?")
    print(result)



