import faiss
import numpy as np
import json
import os
from langchain_openai import OpenAIEmbeddings

# FAISS 인덱스, 메타데이터, 임베딩 모델 초기화 변수
faiss_index = None
doc_metadata = None
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_faiss_and_metadata(index_file="faiss_index.bin", metadata_file="metadata.json"):
    """
    FAISS 인덱스와 메타데이터를 로드합니다.
    """
    global faiss_index, doc_metadata

    if not faiss_index or not doc_metadata:
        # FAISS 인덱스 로드
        if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
            raise FileNotFoundError(f"파일 {index_file} 또는 {metadata_file}가 존재하지 않습니다.")

        faiss_index = faiss.read_index(index_file)

        # 메타데이터 로드
        with open(metadata_file, "r", encoding="utf-8") as f:
            doc_metadata = json.load(f)

    return faiss_index, doc_metadata

def search_with_keywords(keywords, top_k=3):
    """
    키워드별로 검색한 결과를 병합하여 키워드 순환 방식으로 반환합니다.
    :param keywords: 검색 키워드 리스트
    :param top_k: 각 키워드당 반환할 최대 결과 수
    :return: 키워드 순환 방식으로 정렬된 검색 결과 리스트
    """
    # FAISS 및 메타데이터 로드
    faiss_index, doc_metadata = load_faiss_and_metadata()

    # 각 키워드별 검색 결과 저장
    keyword_results = []
    for keyword in keywords:
        query_vector = np.array([embedding_model.embed_query(keyword)]).astype("float32")
        distances, indices = faiss_index.search(query_vector, top_k)

        keyword_result = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(doc_metadata):
                keyword_result.append({"metadata": doc_metadata[idx], "distance": float(dist)})
        keyword_results.append(keyword_result)

    # 키워드 순환 방식으로 결과 병합
    combined_results = []
    for i in range(top_k):
        for keyword_result in keyword_results:
            if i < len(keyword_result):  # 각 키워드의 결과 개수를 초과하지 않도록 확인
                combined_results.append(keyword_result[i])

    return combined_results