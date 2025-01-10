import json
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI

# 데이터 로드 함수
def load_data():
    """
    JSON 파일에서 데이터를 로드
    """
    with open("/Users/hong-gihyeon/Desktop/NotiFYI_BackEND/app/data/*.json", "r") as file:
        return json.load(file)

# 챗봇 질의 함수
from langchain_openai import ChatOpenAI

def ask_chatbot(user_input: str, results: list = [], model_name: str = "gpt-4", temperature: float = 0.7):
    """
    사용자 입력과 검색 결과를 기반으로 GPT 모델의 응답과 추천 게시물 반환
    """
    # context 초기화
    context = (
        "너는 수강신청 알리미 챗봇이야. 게시물을 추천해달라는 요청인지 구분하고, "
        "요청이라면 게시물을 추천해줘.\n\n"
    )
    
    # Document 객체에서 metadata 가져오기
    for doc in results:
        # doc is a Document
        metadata = doc.metadata  # dict 형태
        title = metadata.get("title", "Unknown Title")
        summary = metadata.get("summary", "No summary available")
        url = metadata.get("url", "No URL")

        # context 문장 생성
        context += f"- Title: {title}\n  Summary: {summary}\n  URL: {url}\n\n"
    
    # 사용자 질문 추가
    context += f"User Question: {user_input}\nAnswer:"
    
    # LLM 호출
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm.predict(context)

    # 추천 게시물
    recommended_posts = []
    if "**게시물을 추천합니다**" in response:
        for doc in results:
            metadata = doc.metadata
            recommended_posts.append({
                "title": metadata.get("title", "Unknown Title"),
                "summary": metadata.get("summary", "No summary available"),
                "url": metadata.get("url", "No link available")
            })

    return response, recommended_posts