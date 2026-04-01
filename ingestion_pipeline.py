import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# ── All paths are relative to THIS file's location, not the working directory ──
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH       = os.path.join(BASE_DIR, "docs")
PERSIST_DIR     = os.path.join(BASE_DIR, "db", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    print("Loading embedding model…")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_path=DOCS_PATH):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"\nDocs folder not found: {docs_path}\n"
            "Make sure 'knowledgebase/docs/' exists next to ingestion_pipeline.py "
            "and contains your .txt or .pdf files."
        )

    documents = []

    # .txt files
    txt_loader = DirectoryLoader(
        path=docs_path, glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    documents.extend(txt_loader.load())

    # .pdf files
    for fname in os.listdir(docs_path):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, fname))
            documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in: {docs_path}\n"
            "Please add your documents there."
        )

    print(f"Loaded {len(documents)} document(s) from '{docs_path}'.")
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory=PERSIST_DIR):
    emb = get_embedding_model()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store saved → '{persist_directory}' ({len(chunks)} chunks).")
    return vs


def ingest(docs_path=DOCS_PATH, persist_directory=PERSIST_DIR, force=False):
    """
    Full ingestion pipeline.
    Skips re-ingestion if the store already exists (set force=True to override).
    """
    print("=== Ingestion Pipeline ===")
    print(f"Docs path : {docs_path}")
    print(f"Vector DB : {persist_directory}\n")

    if os.path.exists(persist_directory) and not force:
        print("Existing vector store found — skipping re-ingestion.")
        emb = get_embedding_model()
        vs = Chroma(
            persist_directory=persist_directory,
            embedding_function=emb,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"Loaded store: {vs._collection.count()} chunks.\n")
        return vs

    documents = load_documents(docs_path)
    chunks    = split_documents(documents)
    vs        = create_vector_store(chunks, persist_directory)
    print("Ingestion complete!\n")
    return vs


if __name__ == "__main__":
    ingest()