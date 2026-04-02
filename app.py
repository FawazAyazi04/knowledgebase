"""
KnowledgeBase Search Engine — Streamlit App
- Enter key submits (st.chat_input)
- Conversational paragraph responses
- Persistent conversation history in sidebar
"""

import os
import json
import uuid
import datetime
import streamlit as st

try:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]  # For Streamlit Cloud
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")   # For local
st.set_page_config(
    page_title="KnowledgeBase Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Manrope:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background-color: #0f0f0f; color: #e2e2dc; }

[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #bbb !important; }

/* Hide default chat input styling overrides */
[data-testid="stChatInput"] textarea {
    background: #111 !important;
    color: #e2e2dc !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #e8a000 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: #e8a000 !important;
    border-radius: 6px !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* User message bubble */
.user-bubble {
    background: #161616;
    border-left: 3px solid #e8a000;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0 6px 60px;
    line-height: 1.65;
}

/* Assistant message bubble */
.ai-bubble {
    background: #111;
    border-left: 3px solid #3d87f5;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 60px 6px 0;
    line-height: 1.8;
    font-size: 0.97rem;
}

.msg-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin-bottom: 7px;
    opacity: 0.6;
}
.user-label { color: #e8a000; }
.ai-label   { color: #3d87f5; }

/* Source chips */
.chips-row { margin-top: 10px; }
.chip {
    display: inline-block;
    background: #1a1a1a; border: 1px solid #2c2c2c;
    color: #666; font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; padding: 2px 8px;
    border-radius: 3px; margin: 2px 3px 0 0;
}

/* Header */
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem; font-weight: 600; color: #f5f5ef;
    border-bottom: 2px solid #e8a000;
    padding-bottom: 5px; margin-bottom: 2px;
}
.app-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #444; margin-bottom: 24px;
}

/* Sidebar conversation buttons */
.stButton > button {
    background: transparent !important;
    color: #999 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 4px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.8rem !important;
    text-align: left !important;
    padding: 6px 10px !important;
    font-weight: 400 !important;
}
.stButton > button:hover {
    border-color: #e8a000 !important;
    color: #e2e2dc !important;
    background: #141400 !important;
}

.conv-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #333;
    margin: -4px 0 8px 4px;
}

/* New chat button special style */
.new-chat-btn > button {
    background: #e8a000 !important;
    color: #0a0a0a !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: .04em !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 80px 20px;
    color: #333;
    font-family: 'JetBrains Mono', monospace;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-text { font-size: 0.85rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_MODEL  = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ── Conversation persistence ───────────────────────────────────────────────────
def _hpath(cid):      return os.path.join(HISTORY_DIR, f"{cid}.json")

def list_conversations():
    out = []
    for f in os.listdir(HISTORY_DIR):
        if not f.endswith(".json"): continue
        try:
            with open(os.path.join(HISTORY_DIR, f)) as fh:
                out.append(json.load(fh))
        except Exception: pass
    return sorted(out, key=lambda x: x.get("updated_at",""), reverse=True)

def save_conv(conv):
    with open(_hpath(conv["id"]), "w") as f:
        json.dump(conv, f, indent=2)

def new_conv():
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "messages": [],
    }

def make_title(q):
    w = q.strip().split()
    return " ".join(w[:7]) + ("…" if len(w) > 7 else "")

def rebuild_lc(messages):
    from langchain_core.messages import HumanMessage, AIMessage
    h = []
    for m in messages:
        if m["role"] == "user":      h.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant": h.append(AIMessage(content=m["content"]))
    return h

def switch_to(conv):
    st.session_state.active_conv = conv
    st.session_state.lc_history  = rebuild_lc(conv["messages"])

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {"active_conv": new_conv(), "lc_history": [], "db": None, "llm": None, "k": 4}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load resources once ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def load_db():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=emb,
                  collection_metadata={"hnsw:space": "cosine"})

@st.cache_resource(show_spinner="Loading LLM…")
def load_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0.3)

try:
    if st.session_state.db  is None: st.session_state.db  = load_db()
    if st.session_state.llm is None: st.session_state.llm = load_llm()
    ready = True
except Exception as e:
    ready = False; startup_err = str(e)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Conversations")
    st.divider()

    with st.container():
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True):
            st.session_state.active_conv = new_conv()
            st.session_state.lc_history  = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    convs     = list_conversations()
    active_id = st.session_state.active_conv["id"]

    for c in convs:
        is_active = c["id"] == active_id
        label = ("▶  " if is_active else "") + c.get("title", "Untitled")
        ts    = c.get("updated_at","")[:16].replace("T"," ")
        if st.button(label, key=f"c_{c['id']}", use_container_width=True):
            switch_to(c); st.rerun()
        st.markdown(f'<div class="conv-date">{ts}</div>', unsafe_allow_html=True)

    if convs:
        st.divider()
        if st.button("🗑  Delete this chat", use_container_width=True):
            p = _hpath(active_id)
            if os.path.exists(p): os.remove(p)
            st.session_state.active_conv = new_conv()
            st.session_state.lc_history  = []
            st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#2a2a2a;'>"
        f"model · {GROQ_MODEL}<br>embed · all-MiniLM-L6-v2<br>vdb · ChromaDB</div>",
        unsafe_allow_html=True,
    )

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔍 KnowledgeBase Search</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">rag · llama-3.1-8b-instant · all-MiniLM-L6-v2 · chromadb</div>', unsafe_allow_html=True)

if not ready:
    st.error(f"Startup failed: {startup_err}")
    st.info("1. Run `python ingestion_pipeline.py`\n2. Set `GROQ_API_KEY` in `.env`")
    st.stop()

conv = st.session_state.active_conv
msgs = conv["messages"]

# ── Render existing messages ───────────────────────────────────────────────────
if not msgs:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📄</div>
        <div class="empty-text">
            Ask anything about your documents.<br>
            I'll answer in plain language and we can go deeper from there.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for m in msgs:
        if m["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">'
                f'<div class="msg-label user-label">You</div>'
                f'{m["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            chips = "".join(
                f'<span class="chip">{os.path.basename(s)}</span>'
                for s in m.get("sources", [])
            )
            # Use st.markdown for AI content so markdown in response renders properly
            st.markdown(
                f'<div class="ai-bubble">'
                f'<div class="msg-label ai-label">Assistant</div>',
                unsafe_allow_html=True,
            )
            st.markdown(m["content"])   # renders as proper markdown / paragraphs
            if chips:
                st.markdown(f'<div class="chips-row">{chips}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ── Chat input — Enter submits natively ───────────────────────────────────────
user_input = st.chat_input("Ask something about your documents…")

if user_input and user_input.strip():
    question = user_input.strip()

    with st.spinner(""):
        try:
            from history_aware_gen import ask_question
            from retrieval_pipeline import retrieve

            answer, new_lc = ask_question(
                user_question=question,
                chat_history=st.session_state.lc_history,
                db=st.session_state.db,
                model=st.session_state.llm,
                k=st.session_state.k,
            )
            st.session_state.lc_history = new_lc

            docs    = retrieve(question, k=st.session_state.k)
            sources = list({d.metadata.get("source","unknown") for d in docs})

            if not msgs:
                conv["title"] = make_title(question)

            conv["messages"].append({"role": "user",      "content": question, "sources": []})
            conv["messages"].append({"role": "assistant",  "content": answer,   "sources": sources})
            conv["updated_at"] = datetime.datetime.now().isoformat()
            save_conv(conv)
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")