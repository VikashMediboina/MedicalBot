from typing import Union
from llama2_inference import doctorOutput
from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    message: str
    other_param: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/items")
def read_item(message: str):
    return doctorOutput(message,"")