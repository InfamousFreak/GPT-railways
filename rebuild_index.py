import os
import faiss
import pickle
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("IndexRebuilder")

def rebuild_faiss_index():
    """
    Connects to MongoDB, reads all documents, generates embeddings,
    and creates a new FAISS index and ID map.
    """
    load_dotenv()

    # --- Database Connection ---
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

    if not all([MONGODB_URI, MONGODB_DBNAME, MONGODB_COLLECTION]):
        logger.error("MongoDB environment variables not set. Exiting.")
        return

    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DBNAME]
        collection = db[MONGODB_COLLECTION]
        # Test connection
        client.admin.command('ping')
        logger.info("✅ MongoDB connected successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        return

    # --- Load Model ---
    try:
        logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = embedder.get_sentence_embedding_dimension()
        logger.info(f"✅ Model loaded. Embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        return

    # --- Fetch Data and Create Embeddings ---
    logger.info(f"Fetching documents from collection '{MONGODB_COLLECTION}'...")
    try:
        documents = list(collection.find({"text": {"$exists": True}, "doc_id": {"$exists": True}}))
        if not documents:
            logger.warning("No documents with 'text' and 'doc_id' fields found in the database. Exiting.")
            return
        
        logger.info(f"Found {len(documents)} documents to index.")
        
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['doc_id'] for doc in documents]

        logger.info("Generating embeddings for all documents... (This may take a while)")
        embeddings = embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        logger.info("✅ Embeddings generated successfully.")

    except Exception as e:
        logger.error(f"❌ Failed to fetch documents or generate embeddings: {e}")
        return

    # --- Build and Save FAISS Index ---
    try:
        logger.info("Building new FAISS index...")
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        logger.info(f"✅ FAISS index built with {index.ntotal} vectors.")

        id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}

        FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
        PKL_PATH = os.path.splitext(FAISS_PATH)[0] + ".pkl"

        # Ensure directory exists
        os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

        logger.info(f"Saving FAISS index to {FAISS_PATH}")
        faiss.write_index(index, FAISS_PATH)

        logger.info(f"Saving ID map to {PKL_PATH}")
        with open(PKL_PATH, "wb") as f:
            pickle.dump(id_map, f)

        logger.info("✅ Successfully rebuilt and saved FAISS index and ID map.")

    except Exception as e:
        logger.error(f"❌ Failed to build or save FAISS index: {e}")
        return

if __name__ == "__main__":
    rebuild_faiss_index()

