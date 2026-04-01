# 🔍 KnowledgeBase Search Engine

A fully free, open-source RAG (Retrieval-Augmented Generation) web app built with Streamlit.

## Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Llama 3 8B via **Groq** | Free tier |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace | Free / local |
| Vector DB | **ChromaDB** | Free / local |
| Frontend | **Streamlit** | Free |

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create an API key
4. Copy it

### 3. Configure environment (optional — can also enter in UI)

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 4. Add your documents

Place `.txt` or `.pdf` files in `knowledgebase/docs/`.  
(You can also upload files directly in the Streamlit sidebar.)

### 5. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                  # Streamlit web app (main entry point)
├── ingestion_pipeline.py   # Load docs → chunk → embed → store in ChromaDB
├── retrieval_pipeline.py   # Query ChromaDB → return relevant chunks
├── answer_generation.py    # Single-turn RAG answer generation
├── history_aware_gen.py    # Multi-turn chat with conversation history
├── requirements.txt
├── .env.example
└── knowledgebase/
    └── docs/               # Place your .txt and .pdf files here
```

---

## How It Works

1. **Ingestion**: Documents are loaded, split into overlapping chunks, embedded using a local HuggingFace model, and stored in ChromaDB.
2. **Retrieval**: User queries are embedded and matched against stored chunks via cosine similarity.
3. **Generation**: Retrieved chunks + conversation history are sent to Llama 3 (via Groq) to synthesize a grounded answer.

---

## Running Scripts Directly

```bash
# Ingest documents
python ingestion_pipeline.py

# Test retrieval only
python retrieval_pipeline.py

# Single-turn answer
python answer_generation.py

# Multi-turn CLI chat
python history_aware_gen.py
```
