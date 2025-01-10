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
def ask_chatbot(user_input: str, model_name:str = "gpt-4o-mini", temperature: float = 0.7):
    """
    사용자 입력을 기반으로 JSON 데이터를 참조하여 답변 생성
    :param user_input: 사용자가 입력한 질문
    :param model_name: 사용할 OpenAI 모델 이름
    :param temperature: 텍스트 생성 온도
    :return: 모델의 답변
    """
    # 데이터 로드
    data = load_data()

    # Context 생성
    context = "Answer the question based on the following JSON data:\n\n"
    for item in data:
         context += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    # 프롬프트 생성
    prompt = f"{context}Q: {user_input}\nA:"

    # LLM 초기화 및 응답 생성
    #llm = get_llm(model_name=model_name, temperature=temperature)
    llm = ChatOpenAI(model="gpt-4o-mini",  temperature=temperature)

    response = llm(prompt)

    return response