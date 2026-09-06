import os

from dotenv import load_dotenv

load_dotenv()

MINERU_API_TOKEN=os.getenv("MINERU_API_TOKEN")
MINERU_BASE_URL=os.getenv("MINERU_BASE_URL")

print(MINERU_API_TOKEN)
print(MINERU_BASE_URL)
