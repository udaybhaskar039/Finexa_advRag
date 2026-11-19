
from langchain_community.vectorstores import FAISS


def create_vector_store(documents, embedding_model):
    if not documents:
        raise ValueError("No documents available to create vector store.")

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    vector_store.save_local("embeddings/finexa_faiss")

    print("FAISS vector database created successfully.")

    return vector_store


def load_vector_store(embedding_model):
    vector_store = FAISS.load_local(
        "embeddings/finexa_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store
