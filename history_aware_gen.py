from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR     = os.path.join(BASE_DIR, "db", "chroma_db")
GROQ_MODEL      = "llama-3.1-8b-instant"

# ── System prompt: conversational, paragraph-style, invites follow-up ─────────
SYSTEM_PROMPT = """You are a knowledgeable, conversational assistant helping a user explore a document collection.

How to respond:
- Always write in flowing paragraphs. Never use bullet points, numbered lists, or headers.
- Sound natural and warm, like a knowledgeable friend explaining something — not a search engine returning results.
- Use only the information from the provided documents. If the answer isn't there, say so honestly in one sentence.
- After answering, naturally invite the user to dig deeper — ask a follow-up question, suggest what else they might want to know, or offer to elaborate on something you mentioned. Keep this brief and feel organic, not scripted.
- Keep responses concise but complete. Aim for 3–5 sentences for simple questions, a short paragraph or two for complex ones.
- Never start your response with "Based on the documents" or "According to the provided documents" — just answer directly."""


def get_db():
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=emb,
        collection_metadata={"hnsw:space": "cosine"},
    )


def rewrite_question(model, chat_history, user_question):
    """Make a follow-up question standalone for retrieval."""
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(content=(
            "Given the conversation history, rewrite the latest question so it is "
            "fully standalone and searchable without the history. "
            "Return ONLY the rewritten question, nothing else."
        )),
    ] + chat_history + [
        HumanMessage(content=f"Latest question: {user_question}")
    ]
    return model.invoke(messages).content.strip()


def ask_question(user_question, chat_history, db, model, k=4):
    """
    Answer a question with conversation history awareness.
    Returns (answer_string, updated_chat_history).
    """
    # Step 1: Standalone question for retrieval
    search_q = rewrite_question(model, chat_history, user_question)

    # Step 2: Retrieve relevant chunks
    docs = db.as_retriever(search_kwargs={"k": k}).invoke(search_q)

    # Step 3: Build prompt
    context = "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    user_prompt = f"""Here is the relevant information from the documents:

{context}

Now answer this question in a natural, conversational way: {user_question}"""

    # Step 4: Get answer with full history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + chat_history + [
        HumanMessage(content=user_prompt)
    ]
    answer = model.invoke(messages).content.strip()

    # Step 5: Update history
    updated_history = chat_history + [
        HumanMessage(content=user_question),
        AIMessage(content=answer),
    ]

    return answer, updated_history


def start_chat():
    """Interactive CLI chat loop."""
    print("=== KnowledgeBase Chat ===\nType 'quit' to exit.\n")
    db      = get_db()
    model   = ChatGroq(model=GROQ_MODEL, temperature=0.3)
    history = []

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not q:
            continue
        answer, history = ask_question(q, history, db, model)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    start_chat()