import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import sys
sys.modules["tensorflow"] = None

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS   # ✅ FAISS instead of Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load env
load_dotenv()

# --- Streamlit page setup ---
st.set_page_config(page_title="📄 RAG Q&A", layout="wide")
st.title("📄 RAG Q&A with Multiple PDFs + Chat History + Summarizer")

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Config")

    # API key
    api_key = st.text_input("Groq API Key", type="password")

    # Model params
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
    chunk_size = st.slider("📏 Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("🔄 Chunk Overlap", 0, 500, 150, step=50)

    # Vectorstore reset (not needed for FAISS but kept optional)
    if st.button("🗑️ Reset Vectorstore"):
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
            st.success("✅ FAISS vectorstore cleared. Please re-upload PDFs.")

    st.caption("Upload PDFs → Ask Questions → Summarize or Chat")

# API key check
if not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
    st.stop()

# --- LLM + Embeddings ---
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", temperature=temperature)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- File upload ---
uploaded_files = st.file_uploader("📚 Upload PDF files", type="pdf", accept_multiple_files=True)
all_docs = []

if uploaded_files:
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = pdf.name
            all_docs.extend(docs)

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")
else:
    st.info("ℹ️ Please upload one or more PDFs to begin.")
    st.stop()

# --- Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(all_docs)

# --- Vectorstore (FAISS) ---
@st.cache_resource
def get_vectorstore(_splits):
    return FAISS.from_documents(_splits, embeddings)   # ✅ replaced Chroma with FAISS

vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

st.sidebar.write(f"🔍 Indexed {len(splits)} chunks into FAISS vectorstore")

# --- History-aware retriever ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase user questions using chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# --- QA chain ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the retrieved context below:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# --- Session state ---
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- Simple summarizer (max 10 sentences) ---
def summarize_pdfs(text: str):
    prompt = f"Summarize the following document into **maximum 10 sentences**:\n\n{text[:12000]}"
    resp = llm.invoke(prompt)
    return resp.content

# --- Main Tabs ---
tab1, tab2 = st.tabs(["💬 Chat Mode", "📑 Summarize Mode"])

# --- Chat Mode ---
with tab1:
    session_id = st.text_input("🆔 Session ID", value="default_session")
    user_q = st.chat_input("💬 Ask your question...")

    if user_q:
        history = get_history(session_id)
        result = conversational_rag.invoke(
            {"input": user_q}, config={"configurable": {"session_id": session_id}}
        )

        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(result["answer"])

        # Retrieved chunks
        if "context" in result:
            with st.expander("📑 Retrieved Chunks"):
                for doc in result["context"]:
                    st.write(f"📄 {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')}):")
                    st.write(doc.page_content[:300] + "…")

        # Chat history
        with st.expander("📖 Chat History"):
            for msg in history.messages:
                role = getattr(msg, "role", msg.type).title()
                st.write(f"**{role}:** {msg.content}")

        # Download chat
        chat_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
        st.download_button("⬇️ Download Chat History", data=chat_text, file_name="chat_history.txt")

# --- Summarize Mode ---
with tab2:
    if st.button("📑 Generate Summary"):
        all_text = "\n\n".join([doc.page_content for doc in all_docs])
        summary = summarize_pdfs(all_text)
        st.subheader("📑 Summary of PDFs")
        st.write(summary)
        st.download_button("⬇️ Download Summary", data=summary, file_name="pdf_summary.txt")
