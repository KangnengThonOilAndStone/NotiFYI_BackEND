import logging
from fastapi import FastAPI, HTTPException
from app.chatbot import ask_chatbot
from app.DB import query_faiss


# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI LangChain Chatbot"}

@app.post("/chat/")

async def chat_with_bot(user_input: str):
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required")
    
    query = "교직원"
    results = query_faiss(faiss_index, query, embedding_model, doc_metadata)
    
    # ask_chatbot 함수 호출
    raw_response = ask_chatbot(user_input)
    
    # content만 추출
    content = raw_response.content if hasattr(raw_response, "content") else "No content available"
    logger.info(f"Response content: {content}")

    
    # content를 JSON으로 반환
    return {"response": content}



#uvicorn app.main:app --reload

#POST http://127.0.0.1:8000/chat/