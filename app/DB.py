from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import faiss
import json
import os
import glob
import numpy as np

def load_data():
    """
    JSON 파일에서 데이터를 로드
    """
    data_dir = "/Users/hong-gihyeon/Desktop/NotiFYI_BackEND/app/data"
    all_data = []

    # 디렉토리 내 모든 JSON 파일 읽기
    for file_path in glob.glob(os.path.join(data_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            # 데이터가 리스트인지 확인 후 병합
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    return all_data

def convert_to_documents(data):
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
    # 기존 인덱스 파일이 있는지 확인
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

    # 인덱스 저장
    faiss.write_index(index, index_file)
    with open(index_file.replace(".bin", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print(f"총 {index.ntotal}개의 벡터가 인덱스에 추가되었습니다.")
    return index, metadata



def query_faiss(index, query, embedding_model, metadata, top_k=2):
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({"metadata": metadata[idx], "distance": dist})
    return results


# 메인 실행 코드
data = load_data()
print(f"로드된 데이터 개수: {len(data)}")
for idx, item in enumerate(data):
    print(f"데이터 {idx}: {type(item)} - {item}")

#print("Loaded data:", data[:2])

documents = convert_to_documents(data)
print(f"변환된 Document 개수: {len(documents)}")

#print("Converted documents:", documents[:2])

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
faiss_index, doc_metadata = create_faiss_index(documents, embedding_model)

# FAISS 인덱스 생성 또는 불러오기
faiss_index, doc_metadata = create_faiss_index(documents, embedding_model, index_file="faiss_index.bin")



query = "교직원"
results = query_faiss(faiss_index, query, embedding_model, doc_metadata)

print("검색 결과:")
for result in results:
    print("유사도:", result["distance"])
    print("메타데이터:", result["metadata"])
    print("---")