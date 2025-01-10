from fastapi import FastAPI, HTTPException, Query
import logging
import os

from app.DB import (
    load_data,
    convert_to_documents,
    create_faiss_vectorstore,
    query_faiss_vectorstore
)
from langchain.embeddings import OpenAIEmbeddings
from app.chatbot import ask_chatbot
from app.home import load_and_shuffle_data
from app.keyword_serch import search_with_keywords

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# 전역 변수: 서버 시작 시 한 번만 생성
VECTORSTORE = None

# OpenAI Embeddings 초기화 (OpenAI API 키 필요)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")


@app.on_event("startup")
async def on_startup():
    """
    서버 시작 시 FAISS Vector Store 생성 (메모리 상) 
    """
    global VECTORSTORE
    try:
        logger.info("서버 시작: 데이터 로드 및 벡터 스토어 초기화...")
        data = load_data(data_dir="app/data")
        documents = convert_to_documents(data)
        VECTORSTORE = create_faiss_vectorstore(documents, embedding_model)
        logger.info("벡터 스토어 초기화 완료.")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {e}")
        VECTORSTORE = None


@app.get("/home/")
async def get_home_data():
    """
    홈 화면 데이터를 랜덤으로 섞어 반환합니다.
    """
    try:
        data = load_and_shuffle_data()
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/chat/")
async def chat_with_bot(user_input: str):
    """
    사용자 질의에 대한 챗봇 응답 및 FAISS 검색 결과 반환
    """
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required")

    # 검색 수행
    if VECTORSTORE is None:
        logger.error("Vector Store가 초기화되지 않았습니다.")
        raise HTTPException(status_code=500, detail="Vector Store is not initialized.")

    try:
        results = query_faiss_vectorstore(VECTORSTORE, user_input, top_k=2)
        logger.info(f"FAISS 검색 결과: {results}")
    except Exception as e:
        logger.error(f"FAISS 검색 중 오류 발생: {e}")
        results = []

    # 챗봇 호출
    try:
        raw_response, recommended_posts = ask_chatbot(user_input, results)
        logger.info(f"챗봇 응답: {raw_response}")
        logger.info(f"추천 게시물: {recommended_posts}")

        return {
            "response": raw_response,
            "faiss_results": results,
            "recommended_posts": recommended_posts
        }
    except Exception as e:
        logger.error(f"챗봇 호출 중 오류 발생: {e}")
        return {
            "response": "챗봇 응답을 생성하지 못했습니다.",
            "faiss_results": results,
            "recommended_posts": []
        }


@app.get("/search/")
async def search_endpoint(keyword: str = Query(..., description="검색 키워드 (예: 학사,장학)")):
    """
    검색 API 엔드포인트
    """
    try:
        keywords = keyword.split(",")
        if not keywords:
            raise HTTPException(status_code=400, detail="키워드가 제공되지 않았습니다.")

        # 키워드 검색
        search_results = search_with_keywords(keywords)
        return {"status": "success", "results": search_results}
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return {"status": "error", "message": str(e)}