import faiss
import numpy as np
import json
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
    키워드를 기반으로 FAISS 인덱스에서 검색합니다.
    :param keywords: 검색 키워드 리스트
    :param top_k: 반환할 최대 결과 수
    :return: 검색 결과 리스트
    """
    # FAISS 및 메타데이터 로드
    faiss_index, doc_metadata = load_faiss_and_metadata()

    results = []
    for keyword in keywords:
        # 임베딩 생성
        query_vector = np.array([embedding_model.embed_query(keyword)]).astype("float32")

        # FAISS 검색
        distances, indices = faiss_index.search(query_vector, top_k)

        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(doc_metadata):
                result = {"metadata": doc_metadata[idx], "distance": dist}
                results.append(result)

    # 결과 정렬 (거리 기준 오름차순)
    results = sorted(results, key=lambda x: x["distance"])
    return results