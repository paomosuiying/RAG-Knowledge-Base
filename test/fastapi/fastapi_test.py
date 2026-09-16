import uvicorn
from fastapi import FastAPI
from openai import BaseModel

app = FastAPI()

class Item(BaseModel):
    name : str
    price : float
    is_offer : bool = None

@app.post("/items")
def create_item(item: Item):
    print("后端接口被访问.....")
    print(item.name)
    print(item.price)
    print(item.is_offer)
    return item


@app.get("/read_root")
def read_root():
    print("read_root后端接口被访问.....")
    print("read_root后端接口被访问.....")
    print("read_root后端接口被访问.....")
    return {"hello" : "world"}

@app.get("/item/{item_id}")
def read_item(item_id: int, q: str):
    print("read_item后端有参数入口被访问")
    return {"item_id": item_id, "q": q}




if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)