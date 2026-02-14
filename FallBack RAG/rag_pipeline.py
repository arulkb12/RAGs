import os, tempfile, requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

INDEX_PATH = "faiss_index"

# ----------------------------------
# Build Vector Store
# ----------------------------------
def build_vectorstore(uploaded_files):
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    with tempfile.TemporaryDirectory() as tmp:
        for file in uploaded_files:
            path = os.path.join(tmp, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            pages = PyPDFLoader(path).load()
            docs.extend(splitter.split_documents(pages))

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# ----------------------------------
# Primary RAG (Docs)
# ----------------------------------
def primary_rag(question, vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        """
Answer ONLY using the context.
If the context is insufficient, say "NOT_ENOUGH_CONTEXT".

Context:
{context}

Question:
{question}
"""
    )

    answer = (prompt | llm).invoke(
        {"context": context, "question": question}
    ).content

    return answer, context

# ----------------------------------
# Web Search via SerpAPI
# ----------------------------------
def serpapi_search(query, num_results=5):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    snippets = []
    for r in data.get("organic_results", []):
        snippet = r.get("snippet")
        if snippet:
            snippets.append(snippet)

    return "\n\n".join(snippets)

# ----------------------------------
# Web RAG (Fallback)
# ----------------------------------
def web_rag(question):
    web_context = serpapi_search(question)

    prompt = ChatPromptTemplate.from_template(
        """
You are a knowledgeable assistant.
Use the web search information to answer clearly.

Web Context:
{context}

Question:
{question}
"""
    )

    return (prompt | llm).invoke(
        {"context": web_context, "question": question}
    ).content

# ----------------------------------
# RAG Router (Docs → Web)
# ----------------------------------
def rag_with_web_fallback(question, vectorstore):
    primary_answer, context = primary_rag(question, vectorstore)

    if (
        "NOT_ENOUGH_CONTEXT" in primary_answer
        or len(context.strip()) < 200
    ):
        return web_rag(question), "Web RAG (SerpAPI)"

    return primary_answer, "Document RAG"