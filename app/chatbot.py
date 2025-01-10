import json
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key)

def load_data():
    with open("app/data/data.json", "r") as file:
        return json.load(file)


def ask_chatbot(user_input: str):
    data = load_data()
    context = "Answer the question based on the following JSON data:\n\n"
    
    for item in data:
        context += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    
    prompt = f"{context}Q: {user_input}\nA:"
    response = llm(prompt)
    return response