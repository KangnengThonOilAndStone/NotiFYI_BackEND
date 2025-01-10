from app.DB import query_faiss

faiss_index, doc_metadata = create_faiss_index(documents, embedding_model, index_file="faiss_index.bin")
