from fastapi import FastAPI, HTTPException
from app.chatbot import ask_chatbot


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI LangChain Chatbot"}

@app.post("/chat/")
async def chat_with_bot(user_input: str):
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required")
    response = ask_chatbot(user_input)
    return {"response": response}