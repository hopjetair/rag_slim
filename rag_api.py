import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

import psycopg2
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

# === Load environment ===
load_dotenv()

# === Constants ===
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_PDF_PATH = "./data/Airline_Regulations_v1.0.pdf" 
CONNECTION_STRING = "postgresql://hopjetair:SecurePass123!@localhost:5432/hopjetairline_db" #"postgresql://ppm:airlinerag@localhost:5432/ragdb"
COLLECTION_NAME = "airline_docs_pg"

# === FastAPI Setup ===
app = FastAPI(title="RAG PDF API")

# === Request/Response Schemas ===
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

# === Utility: Check if collection exists ===
def check_pgvector_collection_exists(conn_str, collection_name):
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM langchain_pg_collection WHERE name = %s", (collection_name,))
    exists = cur.fetchone()[0] > 0
    cur.close()
    conn.close()
    return exists

# === Load Vector Store ===
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

collection_exists = check_pgvector_collection_exists(CONNECTION_STRING, COLLECTION_NAME)

if not collection_exists:
    if not os.path.exists(LOCAL_PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {LOCAL_PDF_PATH}")

    loader = PyMuPDFLoader(LOCAL_PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectordb = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING
    )
else:
    vectordb = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING
    )

# === Build RAG Chain ===
retriever = vectordb.as_retriever()
llm = ChatOllama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === Endpoint ===
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    try:
        result = qa_chain.invoke({"query": request.query})
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Run with: uvicorn rag_api:app --reload ===
