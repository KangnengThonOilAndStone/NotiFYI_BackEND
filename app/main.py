from fastapi import FastAPI, HTTPException
from app.chatbot import ask_chatbot
from app.query import query_faiss
import logging
from app.DB import initialize_db, create_faiss_index, load_data, convert_to_documents  # DB 관련 함수 가져오기
from langchain_openai import OpenAIEmbeddings
import os
from app.home import load_and_shuffle_data
from app.keyword import search_with_keywords  # 검색 로직 가져오기



# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# FAISS 인덱스, 임베딩 모델, 메타데이터 초기화 변수
faiss_index = None
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
doc_metadata = None


def delete_files():
    """
    기존 FAISS 인덱스와 메타데이터 파일 삭제
    """
    files_to_delete = ["faiss_index.bin", "metadata.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"{file} 파일 삭제 완료.")
        else:
            logger.info(f"{file} 파일이 존재하지 않습니다.")


# 서버 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 파일 삭제 및 DB 초기화
    """
    global faiss_index, doc_metadata, embedding_model

    try:
        # 기존 파일 삭제
        logger.info("기존 인덱스 및 메타데이터 파일 삭제 중...")
        delete_files()

        # 데이터 로드 및 초기화
        logger.info("새로운 DB 초기화 중...")
        faiss_index, doc_metadata, embedding_model = initialize_db()
        logger.info("DB 초기화 완료.")
    except Exception as e:
        logger.error(f"서버 시작 시 DB 초기화 실패: {e}")
        faiss_index, doc_metadata = None, None

@app.get("/home/")
async def get_home_data():
    """
    홈 화면 데이터를 랜덤으로 섞어 반환합니다.
    """
    try:
        # 데이터 로드 및 섞기
        data = load_and_shuffle_data()
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/chat/")
async def chat_with_bot(user_input: str):
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required")

    if not faiss_index or not doc_metadata or not embedding_model:
        raise HTTPException(status_code=500, detail="DB가 초기화되지 않았습니다.")

    try:
        # FAISS 검색 실행
        results = query_faiss(faiss_index, user_input, embedding_model, doc_metadata)
        logger.info(f"FAISS 검색 결과: {results}")
    except Exception as e:
        logger.error(f"FAISS 검색 중 오류 발생: {e}")
        results = []

    # 챗봇 호출
    raw_response = ask_chatbot(user_input, results)
    content = raw_response.content if hasattr(raw_response, "content") else "No content available"

    return {"response": content, "faiss_results": results}

@app.get("/search/")
async def search_endpoint(keyword: str = Query(..., description="검색 키워드 (예: 학사,장학)")):
    """
    검색 API 엔드포인트
    """
    try:
        # 키워드 처리
        keywords = keyword.split(",")
        if not keywords:
            raise HTTPException(status_code=400, detail="키워드가 제공되지 않았습니다.")

        # FAISS 검색
        search_results = search_with_keywords(keywords)
        return {"status": "success", "results": search_results}
    except FileNotFoundError as fnfe:
        logger.error(f"FAISS 인덱스 또는 메타데이터 파일이 누락되었습니다: {fnfe}")
        return {"status": "error", "message": "FAISS 인덱스 또는 메타데이터 파일이 누락되었습니다."}
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return {"status": "error", "message": str(e)}

#uvicorn app.main:app --reload

#uvicorn app.main:app --reload

#POST http://127.0.0.1:8000/chat/