from fastapi import FastAPI
from pydantic import BaseModel
from src.Rag import RAG
import uvicorn

app = FastAPI()
rag: RAG = RAG()

class Query(BaseModel):
    question: str

@app.get("/get_response")
async def get_response(payload: Query):
    global rag
    result = rag.answer(payload.question, k=5)
    return result.model_dump(mode="json")

if __name__ == "__main__":
    try:
        config = uvicorn.Config(app)
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        print(e)
