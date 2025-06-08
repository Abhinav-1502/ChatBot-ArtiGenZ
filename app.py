from typing import List
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from services.initiator import handle_user_input
from pydantic import BaseModel
import os

app = FastAPI()

#Read and split allowed origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

print(origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Replace with specific origins in production
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

    # ✅ Refresh session after 20 user questions
    user_messages = [msg for msg in chat_history if "user" in msg]
    if len(user_messages) >= 20:
        chat_history = []  # Clear the chat history
        session_store[session_id] = chat_history

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


@api_router.delete("/chat/history")
async def delete_history(request: Request):
    jsonBody = await request.json()
    session_id = jsonBody.get("session_id", "")
    if (session_id == ""):
        return {
            "Please inlcude session_id in the body"
        }
    if session_store.get(session_id, 0) == 0:
        return {
            "session doesn't exist"
        }
    
    session_store.pop(session_id)

    return {
        f"sesssion with id {session_id} deleted successfully"
    }

app.include_router(api_router)