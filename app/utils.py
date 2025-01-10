import os
import glob
import json
from langchain.schema import Document

def load_data(data_dir="data"):
    """
    JSON 파일에서 데이터를 로드
    """
    all_data = []
    for file_path in glob.glob(os.path.join(data_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    return all_data

def convert_to_documents(data):
    """
    데이터를 LangChain Document 객체로 변환
    """
    documents = []
    for item in data:
        doc = Document(
            page_content=item.get("content", ""),
            metadata={
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "target": item.get("target", ""),
                "application": item.get("application", ""),
                "summary": item.get("summary", ""),
                "contact": item.get("contact", "")
            }
        )
        documents.append(doc)
    return documents