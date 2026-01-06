import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🎓 College Assistant Chatbot",
    layout="wide"
)

DATA_DIR = "data"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.npy"

# ---------------- GEMINI ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.chat-user {
    background:#DCF8C6;
    padding:12px;
    border-radius:12px;
    margin:6px 0;
}
.chat-bot {
    background:#F1F0F0;
    padding:12px;
    border-radius:12px;
    margin:6px 0;
}
.card {
    background:#ffffff;
    padding:18px;
    border-radius:16px;
    box-shadow:0 4px 14px rgba(0,0,0,0.12);
    margin-bottom:15px;
}
footer {
    text-align:center;
    margin-top:30px;
    color:#777;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- HELPERS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss():
    index = faiss.read_index(INDEX_FILE)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
    return index, chunks

def gemini_answer(prompt):
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return res.text

def rag_answer(question):
    index, chunks = load_faiss()
    embedder = load_embedder()

    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k=3)

    context = ""
    for idx in I[0]:
        context += chunks[idx]["text"] + "\n"

    if len(context.strip()) < 50:
        return None

    prompt = f"""
Answer using ONLY the context below.
If not found, say: Information not available in documents.

Context:
{context}

Question:
{question}
"""
    return gemini_answer(prompt)

def summarize_pdf(file):
    reader = PdfReader(file)
    text = " ".join([p.extract_text() or "" for p in reader.pages])

    prompt = f"""
Summarize the following document clearly in bullet points:

{text[:8000]}
"""
    return gemini_answer(prompt)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Options")

    gemini_mode = st.checkbox("🌐 Ask Gemini (Outside PDFs)")
    summary_mode = st.checkbox("📄 PDF Summary Mode")

    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []

# ---------------- MAIN ----------------
st.markdown("<h1>🎓 College Assistant Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='card'>Ask questions from college PDFs or general knowledge.</div>", unsafe_allow_html=True)

question = st.text_input("💬 Enter your question")

if question:
    answer = None

    if summary_mode and uploaded:
        answer = summarize_pdf(uploaded)
    else:
        if not gemini_mode:
            answer = rag_answer(question)

        if answer is None:
            answer = gemini_answer(question)

    st.session_state.chat.insert(0, {
        "user": question,
        "bot": answer
    })

# ---------------- CHAT (LATEST ON TOP) ----------------
for chat in st.session_state.chat:
    st.markdown(f"<div class='chat-user'>👤 {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bot'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<footer>
🚀 Developed by <b>BATHALA BALAJI</b> 💻🔥
</footer>
""", unsafe_allow_html=True)
