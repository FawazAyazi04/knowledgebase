# 📚 KnowledgeBase Search Engine: RAG-Powered Document Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)

![Final Website link](https://knowledgebase-syedfawazayazi.streamlit.app/)

## 📌 Project Overview
A modular, full-stack Retrieval-Augmented Generation (RAG) application designed to ingest complex text and PDF documents and synthesize highly accurate, context-aware answers. This project demonstrates how to build a production-ready AI search engine using entirely open-weight models and cost-effective infrastructure.

**Key Features:**
* **Semantic Search:** Replaces rigid keyword matching with dense vector retrieval.
* **History-Aware Generation:** Maintains conversational context for multi-turn follow-up queries.
* **Privacy-First Embeddings:** Document chunking and vector generation occur 100% locally.
* **Low-Latency Inference:** Utilizes the Groq LPU inference engine for near-instantaneous LLM responses.

---

## 🏗️ System Architecture & Tech Stack

| Component | Technology | Rationale / Design Decision |
| :--- | :--- | :--- |
| **LLM Synthesis** | Llama 3 (8B) via **Groq** | Chosen for blazing-fast inference speeds and high-quality reasoning on free-tier constraints. |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Lightweight, CPU-friendly open-source model perfect for high-accuracy local vectorization. |
| **Vector Database** | **ChromaDB** | Local, persistent vector store that removes the need for complex cloud database provisioning. |
| **Frontend UI** | **Streamlit** | Enables rapid prototyping of a clean, responsive chat interface. |
| **Orchestration** | **LangChain** | Manages the complex pipeline of document loading, splitting, retrieving, and prompting. |

---

## ⚙️ How It Works (The RAG Pipeline)

1.  **Ingestion & Processing:** `PyPDFLoader` and `TextLoader` parse raw documents. The text is split using `RecursiveCharacterTextSplitter` with defined chunk overlaps to ensure semantic concepts aren't severed.
2.  **Vectorization:** Text chunks are passed through the local HuggingFace embedding model to generate 384-dimensional vectors, which are persisted to the ChromaDB filesystem.
3.  **Retrieval:** User queries are embedded and matched against the database using Cosine Similarity to extract the top-K most relevant chunks.
4.  **Contextual Synthesis:** The retrieved context and the user's conversation history are injected into a strict system prompt, forcing the Llama 3 model to ground its answers purely in the provided documents.

---

## 📂 Repository Structure

```text
├── app.py                  # Streamlit web app (Main UI entry point)
├── ingestion_pipeline.py   # ETL script: Load docs → chunk → embed → ChromaDB
├── retrieval_pipeline.py   # Core retrieval logic & similarity search implementation
├── answer_generation.py    # Single-turn RAG chain construction
├── history_aware_gen.py    # Multi-turn conversational memory implementation
├── requirements.txt        # Project dependencies
├── .env.example            # Environment variable template
└── knowledgebase/
    └── docs/               # Source directory for .txt and .pdf files