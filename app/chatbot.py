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
def ask_chatbot(user_input: str, results: list = [], model_name: str = "gpt-4", temperature: float = 0.7):
    """
    사용자 입력과 검색 결과를 기반으로 답변 생성
    :param user_input: 사용자가 입력한 질문
    :param results: FAISS 검색 결과 리스트
    :param model_name: 사용할 OpenAI 모델 이름
    :param temperature: 텍스트 생성 온도
    :return: 모델의 답변
    """
    # Context 생성
    context = "너는 수강신청 알리미 챗봇이야. 학생들과 대화와 함께 제공되는 정보를 통해 알맞는 대답을 해줘:\n\n"
    
    for result in results:
        metadata = result.get("metadata", {})
        title = metadata.get("title", "Unknown Title")
        summary = metadata.get("summary", "No summary available")
        context += f"- Title: {title}\n  Summary: {summary}\n\n"
    
    # 사용자 질문 추가
    context += f"User Question: {user_input}\nAnswer:"
    
    # LLM 초기화 및 응답 생성
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm.invoke(context)  # invoke를 사용하여 모델 호출

    return response