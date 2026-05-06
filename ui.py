import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 🔐 API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

st.set_page_config(page_title="AI Resume Assistant", layout="centered")

st.title("🤖 AI Resume Assistant")
st.caption("Upload your resume and get AI-powered feedback instantly 🚀")

# 📁 Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# 💬 Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📄 Process PDF
@st.cache_resource
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(docs, embeddings)

    return db.as_retriever()

# 🚀 Main logic
if uploaded_file:
    retriever = process_pdf(uploaded_file)

    query = st.chat_input("Ask something about your resume...")

    if query:
        st.session_state.chat_history.append(("user", query))

        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an expert resume reviewer.

        Analyze the resume and:
        - Give clear answers
        - Suggest improvements
        - Be specific and professional

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        answer = response.content

        st.session_state.chat_history.append(("ai", answer))

    # 💬 Show chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)