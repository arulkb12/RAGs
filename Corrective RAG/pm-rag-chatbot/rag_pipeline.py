import os
import tempfile
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --------------------
# Setup
# --------------------
load_dotenv()
INDEX_PATH = "faiss_index"

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# --------------------
# Build Vector Store (PDF + TXT)
# --------------------
def build_vectorstore(uploaded_files):
    documents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in uploaded_files:
            file_path = os.path.join(tmp_dir, file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            # ---- Loader selection ----
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue

            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore

# --------------------
# Create RAG Components
# --------------------
def create_rag_components(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    answer_prompt = ChatPromptTemplate.from_template(
        """
You are a Program Management assistant.
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""
    )

    critique_prompt = ChatPromptTemplate.from_template(
        """
You are validating an AI-generated answer.

Question:
{question}

Answer:
{answer}

If the answer is incomplete, vague, or not grounded in context,
reply with exactly: FAIL
Otherwise reply with exactly: PASS
"""
    )

    return retriever, answer_prompt, critique_prompt

# --------------------
# Run Corrective RAG
# --------------------
def run_rag(question, retriever, answer_prompt, critique_prompt):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer_chain = answer_prompt | llm
    answer = answer_chain.invoke(
        {"context": context, "question": question}
    ).content

    critique_chain = critique_prompt | llm
    critique = critique_chain.invoke(
        {"question": question, "answer": answer}
    ).content

    if "FAIL" in critique:
        refined_question = (
            f"Answer clearly and strictly using the context: {question}"
        )
        answer = answer_chain.invoke(
            {"context": context, "question": refined_question}
        ).content

    return answer