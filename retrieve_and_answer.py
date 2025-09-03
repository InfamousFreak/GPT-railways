import os
import base64
import logging
from typing import List, Dict, Any
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import faiss

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("retrieve")

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=API_KEY)
MODEL_ID = "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_ID)

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
faiss_index = None
try:
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
except Exception:
    log.warning("Could not read FAISS index (maybe not built yet).")

TOP_K = 4
IMAGE_KEYWORDS = {"image","diagram","photo","picture","figure","chart","graph","देख","तस्वीर","चित्र","कैसा"}

def _encode(q: List[str]) -> np.ndarray:
    return embed_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def _vector_search_faiss(q_vec: np.ndarray, k:int=TOP_K) -> List[int]:
    global faiss_index
    if faiss_index is None:
        return []
    D, I = faiss_index.search(q_vec, k)
    return I[0].tolist()

def _vector_search_atlas(q_vec: np.ndarray, k:int=TOP_K) -> List[Dict]:
    try:
        pipeline = [{"$vectorSearch": {"index":"vector_index","path":"embedding","queryVector": q_vec.tolist(),"numCandidates": 200, "limit": k}}]
        return list(col.aggregate(pipeline))
    except Exception:
        log.exception("Atlas vectorSearch failed")
        return []

def _get_docs_by_ids(ids: List[int]) -> List[Dict]:
    # FAISS indices don't map directly to Mongo IDs unless we store mapping; fallback to text-match search
    # We'll fetch best matches from Atlas as fallback
    return []

def _read_image_as_dataurl(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read()
            return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

def find_relevant(query: str) -> (List[str], List[str]):
    q_vec = _encode([query])[0]
    # try FAISS
    contexts = []
    if faiss_index is not None:
        try:
            D,I = faiss_index.search(np.expand_dims(q_vec,0), TOP_K)
            # We don't have mapping of faiss idx -> mongo docs in this simple setup,
            # so use Atlas vectorSearch as primary reliable source.
        except Exception:
            log.debug("faiss search failed")
    # use Atlas vectorSearch
    hits = _vector_search_atlas(q_vec, TOP_K*2)
    for h in hits:
        txt = h.get("text_for_search") or h.get("text") or h.get("ocr_text") or ""
        if txt:
            contexts.append(txt[:1200])
    # images
    images = []
    for h in hits:
        if h.get("type") == "image" and h.get("image_path"):
            images.append(_read_image_as_dataurl(h["image_path"]))
    # dedupe contexts
    seen=set()
    out=[]
    for c in contexts:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= TOP_K:
            break
    return out, images[:TOP_K]

def build_prompt(contexts: List[str], query: str) -> str:
    context_text = "\n\n".join(contexts) if contexts else ""
    return (
        "You are a helpful assistant. If the user uses Hindi, answer in Hindi; if they use English, answer in English.\n"
        "Use only the provided context to answer. If you do not have enough information, say so concisely.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )

def generate_answer(prompt: str) -> str:
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        log.exception("Gemini call failed")
        return "I encountered an error while generating the answer."

def answer(question: str) -> Dict[str,Any]:
    if not question or not question.strip():
        return {"answer":"Please ask a non-empty question.", "images":[], "contexts":[]}
    contexts, images = find_relevant(question)
    prompt = build_prompt(contexts, question)
    text = generate_answer(prompt)
    return {"answer": text, "images": images, "contexts": contexts}
