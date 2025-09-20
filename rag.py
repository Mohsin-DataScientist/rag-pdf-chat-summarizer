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
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Streamlit app setup
st.set_page_config(page_title="📚 PDF Chat + Summary", layout="wide")
st.title("📚 PDF Q&A + Summarizer")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Select Model:", ["llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
    temp = st.slider("Temperature:", 0.0, 1.0, 0.3)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# File uploader
uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# ---------------- PDF Processing ----------------
if uploaded_files:
    all_docs = []
    pdf_texts = {}  # store text for each file

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

        # Store raw text per file
        pdf_texts[uploaded_file.name] = "\n\n".join([doc.page_content for doc in docs])

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # ---------------- LLM Setup ----------------
    llm = ChatGroq(model=model_name, api_key=groq_key, temperature=temp)

    # Q/A chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for answering questions from a set of documents."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    parser = StrOutputParser()
    qa_chain = qa_prompt | llm | parser

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------- Summarizer ----------------
    def summarize_pdfs(text: str):
        """Summarize into max 10 lines"""
        prompt = f"Summarize the following document into **maximum 10 sentences**:\n\n{text[:12000]}"
        resp = llm.invoke(prompt)
        summary = resp.content
        return "\n".join(summary.splitlines()[:10])

    # ---------------- Tabs Layout ----------------
    tab1, tab2 = st.tabs(["💬 Chatbot (Q/A)", "📑 Summarizer"])

    # --- Chat Mode ---
    with tab1:
        st.subheader("💬 Ask Questions About Your PDFs")
        user_input = st.text_input("Ask a question:")
        if st.button("🚀 Get Answer"):
            if user_input:
                context = retriever.get_relevant_documents(user_input)
                response = qa_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append(("User", user_input))
                st.session_state.chat_history.append(("Assistant", response))
                st.write("### 🤖 Answer")
                st.write(response)

        if st.session_state.chat_history:
            st.write("### 📜 Chat History")
            for role, msg in st.session_state.chat_history:
                st.write(f"**{role}:** {msg}")

    # --- Summarize Mode ---
    with tab2:
        st.subheader("📑 Summaries of Uploaded PDFs")
        for file_name, text in pdf_texts.items():
            if st.button(f"📑 Summarize {file_name}"):
                summary = summarize_pdfs(text)
                st.write(f"### 📄 {file_name}")
                st.write(summary)
                st.download_button(
                    f"⬇️ Download {file_name} Summary",
                    data=summary,
                    file_name=f"{file_name}_summary.txt"
                )
