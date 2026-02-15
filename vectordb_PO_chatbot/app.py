import streamlit as st
from file_utils import extract_text
from po_pipeline import add_invoice, ask_question

st.title("📦 Structured PO Assistant")

uploaded = st.file_uploader(
    "Upload Invoice (PDF / DOCX / Image)",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

if uploaded:
    if st.button("Process Invoice"):
        with st.spinner("Extracting structured data..."):
            text = extract_text(uploaded)
            data = add_invoice(text)
        st.success("Stored!")
        st.json(data)

st.divider()

question = st.text_input("Ask a question about your invoices")

if st.button("Ask"):
    if question:
        answer = ask_question(question)
        st.write(answer)
