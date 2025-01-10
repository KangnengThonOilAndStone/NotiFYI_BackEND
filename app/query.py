import numpy as np

def query_faiss(index, query, embedding_model, metadata, top_k=3):
    """
    사용자 질의를 통해 FAISS 인덱스에서 가장 유사한 문서를 검색
    """
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({"metadata": metadata[idx], "distance": float(dist)})
    return results