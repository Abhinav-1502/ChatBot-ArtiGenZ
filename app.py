from typing import List
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from services.initiator import handle_user_input
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, etc.
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/finance_chat/api")

## initialize a session store to store chat history
session_store: dict[str, List[dict[str, str]]] = {}

## Defining output structure
class ChatInput(BaseModel):
    session_id: str
    question: str

@api_router.get("/")
def read_root():
    return {"status": "Chatbot API is running"} 



## Chat end point which takes user's question and session ID in body
@api_router.post("/chat")
async def chat(input: ChatInput):
    session_id = input.session_id
    question = input.question

    chat_history = session_store.get(session_id, [])

    try:
        response = await handle_user_input(question, chat_history)
    except Exception as e:
        return {"error": "Internal error processing chat", "details": str(e)}, 500

    chat_history.append({"user": question})
    # chat_history.append({"bot": response["output"]})

    session_store[session_id] = chat_history

    return response

## ✅ New Endpoint: Get Chat History
@api_router.get("/chat/history")
async def get_history(request: Request):
    jsonBody = await request.json()
    history = session_store.get(jsonBody["session_id"])

    if history is None:
        return {"message": "Session not found", "history": []}
    return {"session_id": jsonBody["session_id"], "history": history}

app.include_router(api_router)