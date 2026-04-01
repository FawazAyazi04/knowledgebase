import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "db", "chroma_db")


def get_retriever(k=5, score_threshold=None):
    """
    Build and return a LangChain retriever backed by ChromaDB.

    Args:
        k: Number of chunks to retrieve.
        score_threshold: If set, only return chunks with cosine similarity >= threshold.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    if score_threshold is not None:
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})

    return retriever


def retrieve(query: str, k: int = 5, score_threshold: float = None):
    """
    Retrieve relevant document chunks for a query.

    Returns a list of LangChain Document objects.
    """
    retriever = get_retriever(k=k, score_threshold=score_threshold)
    docs = retriever.invoke(query)
    return docs


if __name__ == "__main__":
    query = "How much did Microsoft pay to acquire GitHub?"
    docs = retrieve(query)

    print(f"Query: {query}\n")
    print("--- Retrieved Chunks ---")
    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i}:\n{doc.page_content}\n")