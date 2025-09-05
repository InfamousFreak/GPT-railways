# app.py
import os
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# MongoDB config
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

if not all([MONGODB_URI, MONGODB_DBNAME, MONGODB_COLLECTION]):
    raise RuntimeError("MongoDB config not set in .env")

# FAISS config
FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")

# Connect MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DBNAME]
collection = db[MONGODB_COLLECTION]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index (create if missing)
if os.path.exists(FAISS_PATH):
    index = faiss.read_index(FAISS_PATH)
    # The pkl file path should match the FAISS_PATH but with .pkl extension
    pkl_path = os.path.splitext(FAISS_PATH)[0] + ".pkl"
    with open(pkl_path, "rb") as f:
        # Assuming the pickle file only contains the id_map
        id_map = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384-dim for MiniLM
    id_map = {}


# FastAPI app
app = FastAPI(title="Industry-Ready RAG API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------- Models -------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ------------------- Routes -------------------
@app.get("/")
async def root():
    """Health check"""
    return {"message": "✅ RAG server running with MongoDB + FAISS!"}


@app.get("/documents")
async def list_documents():
    """List up to 50 ingested documents"""
    docs = collection.find({}, {"_id": 0}).limit(50)
    return {"documents": list(docs)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF/docx/txt).
    - Saves file to ./uploads
    - Real parsing & ingestion handled by `ingest_to_mongodb.py`
    """
    os.makedirs("./uploads", exist_ok=True)
    file_path = os.path.join("./uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc_meta = {
        "filename": file.filename,
        "path": file_path,
        "status": "uploaded",
    }
    collection.insert_one(doc_meta)
    return {"message": f"📄 File {file.filename} uploaded successfully", "metadata": doc_meta}


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the vector database (FAISS + MongoDB)
    """
    query_vec = embedder.encode([request.query]).astype("float32")
    D, I = index.search(query_vec, request.top_k)

    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        mongo_id = id_map.get(int(idx)) # Ensure idx is an integer for dictionary key lookup
        if mongo_id:
            doc = collection.find_one({"doc_id": mongo_id}, {"_id": 0})
            if doc:
                results.append(doc)

    return {"query": request.query, "results": results}


# THE DUPLICATE FUNCTION HAS BEEN REMOVED.


@app.post("/ask")
async def ask(q: QueryRequest):  # same schema as QueryRequest
    # This now correctly calls the one and only query_documents function
    return await query_documents(q)


@app.post("/add_text")
async def add_text_to_index(text: str = Form(...), doc_id: str = Form(...)):
    """
    Add raw text to MongoDB + FAISS index.
    Useful for testing ingestion without PDF.
    """
    # Insert into MongoDB
    doc = {"doc_id": doc_id, "text": text}
    collection.insert_one(doc)

    # Embed + add to FAISS
    vec = embedder.encode([text]).astype("float32")
    idx = index.ntotal
    index.add(vec)
    id_map[idx] = doc_id

    # Save FAISS + mapping
    faiss.write_index(index, FAISS_PATH)
    pkl_path = os.path.splitext(FAISS_PATH)[0] + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(id_map, f)

    return {"message": "✅ Text added successfully", "doc_id": doc_id}
