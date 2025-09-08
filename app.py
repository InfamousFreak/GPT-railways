# app.py - Complete MongoDB Atlas TLS Fix
import os
import faiss
import pickle
import numpy as np
import uvicorn
import ssl
import certifi
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB config
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

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

# Global variables for lazy loading
client = None
db = None
collection = None
embedder = None
index = None
id_map = {}

def create_ssl_context():
    """Create a permissive SSL context for MongoDB Atlas"""
    try:
        # Create SSL context with system certificates
        context = ssl.create_default_context(cafile=certifi.where())
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        # Allow older TLS versions if needed
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        return context
    except Exception as e:
        logger.warning(f"Failed to create custom SSL context: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global client, db, collection, embedder, index, id_map
    
    try:
        # MongoDB setup with comprehensive TLS handling
        if MONGODB_URI and MONGODB_DBNAME and MONGODB_COLLECTION:
            logger.info("Connecting to MongoDB Atlas...")
            
            # Create custom SSL context
            custom_ssl_context = create_ssl_context()
            
            # Comprehensive connection strategies
            connection_configs = [
                # Strategy 1: Custom SSL context (most likely to work)
                {
                    'ssl': True,
                    'ssl_context': custom_ssl_context,
                    'serverSelectionTimeoutMS': 30000,
                    'socketTimeoutMS': 30000,
                    'connectTimeoutMS': 30000,
                    'maxPoolSize': 10,
                    'retryWrites': True
                } if custom_ssl_context else None,
                
                # Strategy 2: Disable all SSL verification
                {
                    'tls': True,
                    'tlsAllowInvalidCertificates': True,
                    'tlsAllowInvalidHostnames': True,
                    'tlsDisableOCSPEndpointCheck': True,
                    'serverSelectionTimeoutMS': 30000,
                    'socketTimeoutMS': 30000,
                    'connectTimeoutMS': 30000
                },
                
                # Strategy 3: Force TLS 1.2
                {
                    'ssl': True,
                    'ssl_cert_reqs': ssl.CERT_NONE,
                    'ssl_check_hostname': False,
                    'ssl_ca_certs': certifi.where(),
                    'serverSelectionTimeoutMS': 30000,
                    'socketTimeoutMS': 30000,
                    'connectTimeoutMS': 30000
                },
                
                # Strategy 4: Minimal TLS options
                {
                    'tls': True,
                    'serverSelectionTimeoutMS': 45000,
                    'socketTimeoutMS': 45000,
                    'connectTimeoutMS': 45000,
                    'heartbeatFrequencyMS': 10000,
                    'maxPoolSize': 1  # Reduce connection pool
                },
                
                # Strategy 5: Legacy SSL approach
                {
                    'ssl': True,
                    'serverSelectionTimeoutMS': 60000,
                    'socketTimeoutMS': 60000,
                    'connectTimeoutMS': 60000
                }
            ]
            
            # Remove None entries
            connection_configs = [config for config in connection_configs if config is not None]
            
            client = None
            for i, config in enumerate(connection_configs):
                try:
                    logger.info(f"Trying connection strategy {i+1}/{len(connection_configs)}")
                    logger.info(f"Strategy {i+1} config: {list(config.keys())}")
                    
                    # Create client with current strategy
                    client = MongoClient(MONGODB_URI, **config)
                    
                    # Test connection with ping
                    logger.info(f"Testing connection with ping...")
                    result = client.admin.command('ping')
                    logger.info(f"Ping successful: {result}")
                    
                    # Test database access
                    db = client[MONGODB_DBNAME]
                    collection = db[MONGODB_COLLECTION]
                    
                    # Test collection access
                    logger.info("Testing collection access...")
                    doc_count = collection.count_documents({}, limit=1)
                    logger.info(f"Collection accessible, sample count: {doc_count}")
                    
                    logger.info(f"✅ MongoDB Atlas connected successfully with strategy {i+1}")
                    break
                    
                except Exception as strategy_error:
                    error_msg = str(strategy_error)
                    logger.warning(f"❌ Strategy {i+1} failed: {error_msg[:200]}...")
                    
                    if client:
                        try:
                            client.close()
                        except:
                            pass
                    client = None
                    db = None
                    collection = None
                    continue
            
            if client is None:
                logger.error("🚨 ALL MongoDB connection strategies failed!")
                logger.error("This might be due to:")
                logger.error("1. IP not whitelisted in MongoDB Atlas")
                logger.error("2. Wrong credentials in MONGODB_URI")
                logger.error("3. Network/firewall issues")
                logger.error("4. MongoDB Atlas cluster issues")
                logger.error("5. TLS/SSL compatibility problems")
        else:
            logger.warning("MongoDB config not complete - running without database")
            logger.info("Required env vars: MONGODB_URI, MONGODB_DBNAME, MONGODB_COLLECTION")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Embedding model loaded successfully")
        
        # FAISS setup
        FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
        
        if os.path.exists(FAISS_PATH):
            logger.info(f"Loading FAISS index from {FAISS_PATH}")
            index = faiss.read_index(FAISS_PATH)
            pkl_path = os.path.splitext(FAISS_PATH)[0] + ".pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    id_map = pickle.load(f)
            logger.info("✅ FAISS index loaded successfully")
        else:
            logger.info("Creating new FAISS index")
            index = faiss.IndexFlatL2(384)  # 384-dim for MiniLM
            id_map = {}
            
    except Exception as e:
        logger.error(f"💥 Startup error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# ------------------- Models -------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# ------------------- Routes -------------------
@app.get("/")
async def root():
    """Health check"""
    return {"message": "✅ RAG server running!", "mongodb_status": "connected" if collection else "disconnected"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    status = {
        "mongodb": "connected" if collection is not None else "disconnected",
        "embedder": "loaded" if embedder is not None else "not loaded",
        "faiss": "loaded" if index is not None else "not loaded",
        "faiss_size": index.ntotal if index is not None else 0,
        "environment_vars": {
            "MONGODB_URI": "set" if MONGODB_URI else "not set",
            "MONGODB_DBNAME": MONGODB_DBNAME if MONGODB_DBNAME else "not set",
            "MONGODB_COLLECTION": MONGODB_COLLECTION if MONGODB_COLLECTION else "not set"
        }
    }
    
    # Try to get MongoDB server info if connected
    if client:
        try:
            server_info = client.server_info()
            status["mongodb_version"] = server_info.get("version", "unknown")
        except:
            pass
    
    return {"status": "healthy", "components": status}

@app.get("/test-connection")
async def test_connection():
    """Test MongoDB connection endpoint"""
    if not client:
        return {"status": "error", "message": "No MongoDB connection"}
    
    try:
        # Test ping
        ping_result = client.admin.command('ping')
        
        # Test database access
        db_stats = db.command("dbstats") if db else None
        
        # Test collection access
        collection_count = collection.count_documents({}) if collection else None
        
        return {
            "status": "success",
            "ping": ping_result,
            "database": MONGODB_DBNAME,
            "collection": MONGODB_COLLECTION,
            "document_count": collection_count,
            "db_stats": db_stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents")
async def list_documents():
    """List up to 50 ingested documents"""
    if collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        docs = list(collection.find({}, {"_id": 0}).limit(50))
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF/docx/txt)"""
    if collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        os.makedirs("./uploads", exist_ok=True)
        file_path = os.path.join("./uploads", file.filename)
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        doc_meta = {
            "filename": file.filename,
            "path": file_path,
            "status": "uploaded",
            "size": len(content)
        }
        collection.insert_one(doc_meta)
        return {"message": f"📄 File {file.filename} uploaded successfully", "metadata": doc_meta}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the vector database (FAISS + MongoDB)"""
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not available")
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not available")
    
    try:
        query_vec = embedder.encode([request.query]).astype("float32")
        D, I = index.search(query_vec, request.top_k)

        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            mongo_id = id_map.get(int(idx))
            if mongo_id and collection:
                doc = collection.find_one({"doc_id": mongo_id}, {"_id": 0})
                if doc:
                    results.append(doc)

        # If no results from FAISS/MongoDB, try direct text search as fallback
        if not results and collection:
            try:
                text_search_results = list(collection.find(
                    {"$text": {"$search": request.query}}, 
                    {"_id": 0}
                ).limit(request.top_k))
                results.extend(text_search_results)
            except:
                pass

        return {"query": request.query, "results": results, "total_found": len(results)}
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to query documents")

@app.post("/ask")
async def ask(request: QueryRequest):
    """Alias for query endpoint"""
    return await query_documents(request)

@app.post("/add_text")
async def add_text_to_index(text: str = Form(...), doc_id: str = Form(...)):
    """Add raw text to MongoDB + FAISS index"""
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not available")
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not available")
    if collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Insert into MongoDB
        doc = {"doc_id": doc_id, "text": text}
        collection.insert_one(doc)

        # Embed + add to FAISS
        vec = embedder.encode([text]).astype("float32")
        idx = index.ntotal
        index.add(vec)
        id_map[idx] = doc_id

        # Save FAISS + mapping
        FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
        os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
        faiss.write_index(index, FAISS_PATH)
        pkl_path = os.path.splitext(FAISS_PATH)[0] + ".pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(id_map, f)

        return {"message": "✅ Text added successfully", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error adding text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add text")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)