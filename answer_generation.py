from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval_pipeline import retrieve
from dotenv import load_dotenv

load_dotenv()

# Free Groq model — fast Llama 3 inference
GROQ_MODEL = "llama-3.1-8b-instant"


def build_prompt(query: str, docs: list) -> str:
    context = "\n\n".join(
        [f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )
    return f"""You are a helpful assistant. Answer the user's question using ONLY the information from the documents below.
If the answer is not contained in the documents, say: "I don't have enough information to answer that based on the provided documents."

Question: {query}

Documents:
{context}

Answer:"""


def generate_answer(query: str, k: int = 5) -> dict:
    """
    Full RAG answer generation for a single query.

    Returns a dict with keys: query, answer, sources
    """
    docs = retrieve(query, k=k)

    if not docs:
        return {
            "query": query,
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
        }

    prompt = build_prompt(query, docs)

    model = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    messages = [
        SystemMessage(content="You are a precise, helpful assistant."),
        HumanMessage(content=prompt),
    ]

    result = model.invoke(messages)
    answer = result.content.strip()

    sources = list(
        {doc.metadata.get("source", "Unknown") for doc in docs}
    )

    return {"query": query, "answer": answer, "sources": sources, "docs": docs}


if __name__ == "__main__":
    query = "How much did Microsoft pay to acquire GitHub?"
    output = generate_answer(query)

    print(f"Query: {output['query']}\n")
    print(f"Answer:\n{output['answer']}\n")
    print(f"Sources: {output['sources']}")