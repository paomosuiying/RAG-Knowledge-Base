import os
from dataclasses import  dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MinIoConfig:
   endpoint :str
   access_key: str
   secret_key : str
   bucket_name : str
   img_dir: str


minio_config = MinIoConfig(
   endpoint=os.getenv("MINIO_ENDPOINT"),# MinIO 服务端点
   access_key=os.getenv("MINIO_ACCESS_KEY"),# 访问密钥
   secret_key=os.getenv("MINIO_SECRET_KEY"),# 私有密钥
   bucket_name=os.getenv("MINIO_BUCKET_NAME"),# 存储桶名称
   img_dir=os.getenv("MINIO_IMG_DIR","")
)
