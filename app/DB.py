import os
import glob
import json
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def load_data(data_dir="app/data"):
    """
    JSON 파일에서 데이터를 로드하여 list에 담아 반환합니다.
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
    로드된 JSON 데이터를 LangChain의 Document 객체 리스트로 변환합니다.
    """
    documents = []
    for item in data:
        # dict 형태가 아닐 시 스킵
        if not isinstance(item, dict):
            print(f"Invalid item skipped: {item}")
            continue
        
        doc = Document(
            page_content=item.get("content", ""),
            metadata={
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "target": item.get("target", ""),
                "application": item.get("application", ""),
                "summary": item.get("summary", ""),
                "contact": item.get("contact", ""),
                "url": item.get("url", "")
            }
        )
        documents.append(doc)
    return documents


def create_faiss_vectorstore(documents, embedding_model):
    """
    Document 리스트와 임베딩 모델을 사용하여 FAISS Vector Store를 생성합니다.
    디스크에 저장하지 않고, 메모리 상에서만 유지합니다.
    """
    print("새로운 벡터스토어를 생성합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    print(f"분할된 Document 개수: {len(split_docs)}")

    # 문서 리스트로부터 FAISS Vector Store 생성
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    return vectorstore


def query_faiss_vectorstore(vectorstore, query, top_k=2):
    """
    Vector Store에 질의(query)하여 상위 top_k개 결과를 반환합니다.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return results