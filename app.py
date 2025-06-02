from fastapi import FastAPI, Request
from services.rag_pipeline import get_rag_response
from services.sql_generator import generate_sql

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Chatbot API is running"}

@app.get("/chat/rag")
async def chat_with_docs(request: Request):
    body = await request.json()
    question = body.get("question")
    return get_rag_response()

@app.post("/chat/sql")
async def generate_query(request: Request):
    body = await request.json()
    question = body.get("question")
    return generate_sql(question)


