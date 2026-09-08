import base64

from config.lm_config import lm_config
from utils.llm_utils import get_llm_client

base64_image = None
with open("F:/output\hak180产品安全手册/images/8e839864036a7326885565163d99117ea943ecd29a656c85e7aa4052a9b9d28d.jpg","rb") as img:
    base64_image = base64.b64encode(img.read()).decode("utf-8")
#1.vlm模型调用
vl_ai = get_llm_client(lm_config.vl_model)

# 2.调用模型
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""这是书籍管理文件中的一张图片，图片上文部分为这是一堆好书"，下文部分为请好好阅读，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

response = vl_ai.invoke(messages)
print(response)