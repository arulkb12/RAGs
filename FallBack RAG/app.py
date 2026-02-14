import streamlit as st
from rag_pipeline import build_vectorstore, rag_with_web_fallback

st.set_page_config(page_title="RAG with Web Fallback")

st.title("📚 Program Management RAG (Web Fallback Enabled)")

uploaded_files = st.file_uploader(
    "Upload Program Management PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        vectorstore = build_vectorstore(uploaded_files)

    st.session_state.vectorstore = vectorstore
    st.success("Documents indexed successfully!")

question = st.text_input("Ask your question")

if question and "vectorstore" in st.session_state:
    with st.spinner("Thinking..."):
        answer, source = rag_with_web_fallback(
            question,
            st.session_state.vectorstore
        )

    st.markdown(f"**Answer Source:** `{source}`")
    st.write(answer)