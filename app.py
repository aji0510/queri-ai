import os
import hashlib

import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(
    page_title="Queri AI",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    :root {
        --olive: #4C5532;
        --olive-dark: #394329;
        --sand: #F3E8D9;
        --sand-light: #F8F1E8;
        --ivory: #FCFAF5;
        --navy: #172B49;
        --brown: #8B6548;
        --muted: #667080;
        --line: #E8E1D7;
    }

    /* Main application */
    .stApp {
        background: var(--ivory);
        color: var(--navy);
    }

    [data-testid="stHeader"] {
        background: rgba(252, 250, 245, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #F2EEE5;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    /* Branding */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.2rem 0 1.5rem 0.1rem;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 13px;
        background: var(--olive);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -2px;
    }

    .brand-name {
        color: var(--navy);
        font-size: 31px;
        font-weight: 900;
        letter-spacing: -1.2px;
        line-height: 1.05;
    }

    .sidebar-label {
        color: var(--navy);
        font-weight: 750;
        font-size: 1.02rem;
        margin: 1.1rem 0 0.35rem;
    }

    .sidebar-muted {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .sidebar-tip {
        background: #E8E8D9;
        border-radius: 14px;
        padding: 14px 15px;
        margin-top: 1.2rem;
        color: #35402B;
        font-size: 0.9rem;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(110deg, #F3E8D9 0%, #F7F0E7 100%);
        border-radius: 18px;
        padding: 30px 36px 27px;
        margin: 0.3rem 0 1.15rem;
        border: 1px solid #F0E7DB;
    }

    .hero-title {
        font-size: clamp(2.7rem, 5vw, 4.25rem);
        font-weight: 950;
        letter-spacing: -2.5px;
        line-height: 1.02;
        color: var(--navy);
        margin: 0 0 0.25rem;
    }

    .hero-subtitle {
        font-size: clamp(1.35rem, 2.4vw, 2rem);
        font-weight: 750;
        letter-spacing: -0.7px;
        color: #303D2B;
        margin-bottom: 0.35rem;
    }

    .hero-description {
        font-size: 1.05rem;
        color: #536071;
        line-height: 1.55;
        max-width: 700px;
    }

    /* Section headings */
    .section-title {
        color: var(--navy);
        font-weight: 850;
        font-size: 1.18rem;
        margin: 0.7rem 0 0.55rem;
    }

    /* Feature cards */
    .feature-card {
        min-height: 160px;
        background: rgba(255, 255, 255, 0.62);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px;
        height: 100%;
    }

    .feature-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        margin-bottom: 14px;
    }

    .icon-olive {
        background: #E8E9D9;
        color: var(--olive);
    }

    .icon-brown {
        background: #F4E5D6;
        color: #704625;
    }

    .icon-blue {
        background: #E3EAF2;
        color: #173D70;
    }

    .icon-sand {
        background: #F6EBDD;
        color: #704625;
    }

    .feature-title {
        color: var(--navy);
        font-weight: 850;
        font-size: 1.03rem;
        margin-bottom: 0.28rem;
    }

    .feature-text {
        color: #657080;
        font-size: 0.93rem;
        line-height: 1.45;
    }

    /* Empty conversation state */
    .empty-chat {
        border: 1px solid var(--line);
        border-radius: 17px;
        background: rgba(255, 255, 255, 0.48);
        text-align: center;
        padding: 26px 16px 24px;
        margin-top: 0.8rem;
        color: var(--muted);
    }

    .empty-chat-icon {
        font-size: 2rem;
        margin-bottom: 0.35rem;
    }

    .empty-chat-title {
        color: var(--navy);
        font-weight: 850;
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
    }

    /* Conversation messages: readable in both themes */
    div[data-testid="stChatMessage"] {
        background: #FFFEFB !important;
        color: #172B49 !important;
        border: 1px solid #E8E1D7 !important;
        border-radius: 15px;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] strong,
    div[data-testid="stChatMessage"] em {
        color: #172B49 !important;
    }

    /* Chat input container */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #E5DED3 !important;
    }

    /* Light theme input text */
    [data-testid="stChatInput"] textarea {
        color: #667080 !important;
        -webkit-text-fill-color: #667080 !important;
        caret-color: #172B49 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #667080 !important;
        -webkit-text-fill-color: #667080 !important;
        opacity: 1 !important;
    }

    /* Dark theme input text.
       Streamlit theme attributes can differ by version. */
    [data-testid="stAppViewContainer"][data-theme="dark"]
    [data-testid="stChatInput"] textarea,
    [data-theme="dark"] [data-testid="stChatInput"] textarea,
    html[data-theme="dark"] [data-testid="stChatInput"] textarea {
        color: #667080 !important;
        -webkit-text-fill-color: #667080 !important;
        caret-color: #667080 !important;
    }

    [data-testid="stAppViewContainer"][data-theme="dark"]
    [data-testid="stChatInput"] textarea::placeholder,
    [data-theme="dark"] [data-testid="stChatInput"] textarea::placeholder,
    html[data-theme="dark"] [data-testid="stChatInput"] textarea::placeholder {
        color: #667080 !important;
        -webkit-text-fill-color: #667080 !important;
        opacity: 1 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid #DCD8C9;
        background: #ECEBDD;
        color: #303D2B;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: var(--olive);
        color: var(--olive-dark);
        background: #E3E5D5;
    }

    /* Expanders and dividers */
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 12px;
    }

    hr {
        border-color: #DED8CD;
    }

    /* Responsive layout */
    @media (max-width: 800px) {
        .hero {
            padding: 23px 21px;
        }

        .brand-name {
            font-size: 27px;
        }

        .hero-title {
            letter-spacing: -1.5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- API setup -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check OPENAI_API_KEY in your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def split_into_chunks(text, page_number, filename):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": page_number, "filename": filename})
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def extract_pdf_chunks(pdf_bytes, filename):
    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text.strip():
                chunks.extend(split_into_chunks(text, page_number, filename))
    return chunks


def create_embeddings(texts):
    all_embeddings = []
    for start in range(0, len(texts), 64):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[start:start + 64],
        )
        all_embeddings.extend(item.embedding for item in response.data)
    embeddings = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def retrieve_chunks(question, chunks, embeddings):
    response = client.embeddings.create(model="text-embedding-3-small", input=question)
    query_vector = np.array(response.data[0].embedding, dtype=np.float32)
    query_vector /= max(np.linalg.norm(query_vector), 1e-12)
    scores = embeddings @ query_vector
    best_indices = np.argsort(scores)[::-1][:TOP_K]
    return [{**chunks[index], "score": float(scores[index])} for index in best_indices]


def answer_question(question, chunks, embeddings, history):
    relevant_chunks = retrieve_chunks(question, chunks, embeddings)
    context = "\n\n".join(
        f"[Source: {item['filename']}, Page: {item['page']}]\n{item['text']}"
        for item in relevant_chunks
    )
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-6:]
    )
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "You are a helpful PDF question-answering assistant. Answer using the retrieved PDF excerpts. "
            "Use conversation history only to understand follow-up questions, not as evidence. "
            "If the excerpts do not contain the answer, say you could not find it in the provided excerpts. "
            "Do not invent facts. Cite the source filename and page number for key claims."
        ),
        input=(
            f"Recent conversation:\n{history_text or 'None'}\n\n"
            f"Retrieved excerpts:\n{context}\n\nCurrent question: {question}"
        ),
    )
    return response.output_text, relevant_chunks


def show_brand():
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">Q</div>
            <div class="brand-name">Queri AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_feature(icon, icon_class, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon {icon_class}">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    show_brand()
    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    st.markdown("💬 **Chat**")
    st.markdown("📄 **Documents**")
    st.markdown("⚙️ **Settings**")
    st.divider()
    st.markdown('<div class="sidebar-label">Your documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Choose one or more PDF documents.",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF(s) selected")
        for uploaded_file in uploaded_files:
            st.markdown(f"📄 {uploaded_file.name}")
    st.markdown(
        """
        <div class="sidebar-tip">
            <strong>ⓘ Multiple PDF support</strong><br>
            Search across all your documents at once.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------- Main UI -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Queri AI</div>
        <div class="hero-subtitle">Find answers in your documents.</div>
        <div class="hero-description">
            Upload your PDFs, ask your questions, and get clear, concise answers
            with source references.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_cols = st.columns(4, gap="medium")
with feature_cols[0]:
    show_feature("▤", "icon-olive", "Multiple PDFs", "Upload and search across all your documents")
with feature_cols[1]:
    show_feature("⌕", "icon-brown", "Smart Search", "Find relevant information instantly")
with feature_cols[2]:
    show_feature("✧", "icon-blue", "Clear Answers", "Get short and precise explanations")
with feature_cols[3]:
    show_feature("▧", "icon-sand", "Source References", "See the document and page number")

st.markdown('<div class="section-title">Try asking</div>', unsafe_allow_html=True)
prompt_cols = st.columns(4, gap="small")
suggestions = [
    ("▤  Summarize documents", "Summarize the key points from the uploaded documents."),
    ("⌕  Explain key concepts", "Explain the key concepts in the uploaded documents."),
    ("☷  Create an MCQ", "Create a multiple-choice question from the uploaded documents and explain the correct answer."),
    ("▧  Find important topics", "List the most important topics covered in the uploaded documents."),
]
for index, (label, prompt) in enumerate(suggestions):
    with prompt_cols[index]:
        if st.button(label, key=f"suggest_{index}", use_container_width=True):
            st.session_state.pending_question = prompt

# ----------------------------- Index PDFs -----------------------------
if uploaded_files:
    file_hashes = sorted(hashlib.sha256(file.getvalue()).hexdigest() for file in uploaded_files)
    collection_hash = hashlib.sha256("|".join(file_hashes).encode()).hexdigest()

    if st.session_state.get("collection_hash") != collection_hash:
        for key in ("collection_hash", "chunks", "embeddings", "messages"):
            st.session_state.pop(key, None)
        try:
            all_chunks = []
            with st.spinner("Reading your PDFs..."):
                for uploaded_file in uploaded_files:
                    all_chunks.extend(extract_pdf_chunks(uploaded_file.getvalue(), uploaded_file.name))
            if not all_chunks:
                st.error("No readable text found in the uploaded PDFs. Scanned documents may require OCR.")
                st.stop()
            with st.spinner("Preparing smart search for your documents..."):
                embeddings = create_embeddings([chunk["text"] for chunk in all_chunks])
            st.session_state.chunks = all_chunks
            st.session_state.embeddings = embeddings
            st.session_state.collection_hash = collection_hash
            st.session_state.messages = []
        except Exception as error:
            st.error("Failed to process the PDFs.")
            st.code(str(error))
            st.stop()

    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    st.caption(f"✓ {len(uploaded_files)} PDF(s) ready · {len(chunks)} text sections indexed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-chat">
                <div class="empty-chat-icon">▱</div>
                <div class="empty-chat-title">Start a conversation</div>
                <div>Ask a question about your uploaded documents.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask a question about your documents...")
    pending_question = st.session_state.pop("pending_question", None)
    question_to_answer = question or pending_question

    if question_to_answer:
        st.session_state.messages.append({"role": "user", "content": question_to_answer})
        with st.chat_message("user"):
            st.markdown(question_to_answer)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    answer, relevant_chunks = answer_question(
                        question_to_answer,
                        chunks,
                        embeddings,
                        st.session_state.messages[:-1],
                    )
                st.markdown(answer)
                with st.expander("View retrieved PDF excerpts"):
                    for item in relevant_chunks:
                        st.markdown(f"**{item['filename']} — Page {item['page']}**")
                        st.write(item["text"])
            except Exception as error:
                answer = "The request failed. Check your API key, billing, internet connection, and model access."
                st.error(answer)
                st.code(str(error))
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.markdown(
        """
        <div class="empty-chat">
            <div class="empty-chat-icon">▱</div>
            <div class="empty-chat-title">Start a conversation</div>
            <div>Upload your documents in the left sidebar and ask anything.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
