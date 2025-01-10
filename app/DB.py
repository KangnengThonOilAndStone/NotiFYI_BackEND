from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import faiss
import json
import os
import glob
import numpy as np


def initialize_db():
    """
    FAISS 인덱스, 메타데이터 및 임베딩 모델 초기화
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index = None
    doc_metadata = None

    if os.path.exists("faiss_index.bin") and os.path.exists("metadata.json"):
        # 기존 인덱스와 메타데이터 로드
        faiss_index = faiss.read_index("faiss_index.bin")
        with open("metadata.json", "r", encoding="utf-8") as f:
            doc_metadata = json.load(f)
        print("기존 인덱스와 메타데이터를 로드했습니다.")
    else:
        # 새로운 데이터 로드 및 처리
        print("새로운 인덱스를 생성합니다.")
        data = load_data()
        documents = convert_to_documents(data)
        vectors = [embedding_model.embed_query(doc.page_content) for doc in documents]

        # 벡터 데이터 생성 및 인덱스 추가
        vectors = np.array(vectors).astype("float32")
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        faiss_index.add(vectors)

        # 인덱스와 메타데이터 저장
        faiss.write_index(faiss_index, "faiss_index.bin")
        with open("metadata.json", "w", encoding="utf-8") as f:
            json.dump([doc.metadata for doc in documents], f)
        doc_metadata = [doc.metadata for doc in documents]

    return faiss_index, doc_metadata, embedding_model


def load_data(data_dir="app/data"):
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
    JSON 데이터를 Document 객체로 변환
    """
    documents = []
    for item in data:
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
                "contact": item.get("contact", "")
            }
        )
        documents.append(doc)
    return documents


def create_faiss_index(documents, embedding_model, index_file="faiss_index.bin"):
    """
    문서에서 임베딩을 생성하여 FAISS 인덱스를 생성하거나 기존 인덱스를 불러옵니다.
    """
    if os.path.exists(index_file):
        print(f"기존 인덱스를 불러옵니다: {index_file}")
        index = faiss.read_index(index_file)
        with open(index_file.replace(".bin", ".meta.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    print("새로운 인덱스를 생성합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    print(f"분할된 Document 개수: {len(split_docs)}")

    vectors = []
    metadata = []
    for doc in split_docs:
        embedding = embedding_model.embed_query(doc.page_content)
        vectors.append(embedding)
        metadata.append(doc.metadata)

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    print(f"생성된 벡터 개수: {len(vectors)}")
    index.add(vectors)

    faiss.write_index(index, index_file)
    with open(index_file.replace(".bin", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"총 {index.ntotal}개의 벡터가 인덱스에 추가되었습니다.")
    return index, metadata


def query_faiss(index, query, embedding_model, metadata, top_k=2):
    """
    사용자의 질의를 통해 FAISS 인덱스에서 가장 유사한 문서 검색
    """
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({"metadata": metadata[idx], "distance": dist})
    return results


def update_faiss_index(new_documents, embedding_model, faiss_index, metadata_file="metadata.json"):
    """
    새로운 문서를 FAISS 인덱스와 메타데이터에 추가
    """
    new_vectors = [embedding_model.embed_query(doc.page_content) for doc in new_documents]
    faiss_index.add(np.array(new_vectors).astype("float32"))

    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.extend([doc.metadata for doc in new_documents])

    faiss.write_index(faiss_index, "faiss_index.bin")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"총 {len(metadata)}개의 문서가 업데이트되었습니다.")
    return metadata


# 메인 실행 코드
if __name__ == "__main__":
    print("데이터 로드 중...")
    data = load_data()
    print(f"로드된 데이터 개수: {len(data)}")

    documents = convert_to_documents(data)
    #rint(f"변환된 Document 개수: {len(documents)}")

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index, doc_metadata = create_faiss_index(documents, embedding_model)
    
    #new_documents = convert_to_documents(data)



    # 테스트용 예제 질의
    query = "교직원"
    results = query_faiss(faiss_index, query, embedding_model, doc_metadata)
    print("검색 결과:")
    for result in results:
        print(f"유사도: {result['distance']}")
        print(f"메타데이터: {result['metadata']}")
        print("---")