from fastapi import FastAPI
from pydantic import BaseModel
from src.Rag import RAG

app = FastAPI()
rag: RAG = RAG()

class Query(BaseModel):
    question: str

@app.get("/get_response")
async def get_response(payload: Query):
    global rag
    result = rag.answer(payload.question, k=5)
    return BaseModel.model_dump(result, mode="json")