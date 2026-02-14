import streamlit as st
from rag_pipeline import (
    build_vectorstore,
    create_rag_components,
    run_rag
)

st.set_page_config(
    page_title="PM Assistant",
    layout="centered"
)

st.title("📊 Program Manager AI Assistant")

# --------------------
# Upload Documents
# --------------------
uploaded_files = st.file_uploader(
    "Upload Program Management documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        vectorstore = build_vectorstore(uploaded_files)
        retriever, answer_prompt, critique_prompt = create_rag_components(vectorstore)

    st.success("Documents indexed successfully. Ask your question 👇")

    st.session_state.retriever = retriever
    st.session_state.answer_prompt = answer_prompt
    st.session_state.critique_prompt = critique_prompt

# --------------------
# Ask Question
# --------------------
if "retriever" in st.session_state:
    question = st.text_input("Ask a Program Management question")

    if question:
        with st.spinner("Thinking..."):
            answer = run_rag(
                question,
                st.session_state.retriever,
                st.session_state.answer_prompt,
                st.session_state.critique_prompt
            )

        st.markdown("### Answer")
        st.write(answer)