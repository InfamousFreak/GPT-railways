import os
import logging
import warnings
import hashlib
import json
from io import BytesIO
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import camelot
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import faiss

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("ingest")

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

PDF_DIR = "./pdf"
OUT_DIR = "./knowledge_pack"
IMG_DIR = os.path.join(OUT_DIR, "images")
FAISS_PATH = os.getenv("FAISS_PATH", os.path.join(OUT_DIR, "index_hnsw.faiss"))
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64
MAX_PAGES = None
MAX_WORKERS = max(1, os.cpu_count() - 1)

# local FAISS (HNSW)
faiss_index = None
faiss_dim = None

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks

def save_image(pil_img: Image.Image, stem: str, page: int, idx: int, quality=70) -> str:
    try:
        pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1600,1600))
        name = f"{stem}_p{page}_i{idx}.jpg"
        path = os.path.join(IMG_DIR, name)
        pil_img.save(path, "JPEG", optimize=True, quality=quality)
        return path
    except Exception:
        log.exception("save_image failed")
        return ""

def extract_tables(pdf_path: str) -> List[Dict]:
    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
        return [t.df.to_dict() for t in tables]
    except Exception as e:
        log.warning("camelot table extraction failed: %s", e)
        return []

def extract_images_and_ocr(pdf_path: str) -> List[Dict]:
    docs = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            img_list = page.get_images(full=True)
            for i, img in enumerate(img_list):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    b = base["image"]
                    with Image.open(BytesIO(b)) as pil_img:
                        if pil_img.width < 60 or pil_img.height < 60:
                            continue
                        img_path = save_image(pil_img, stem, page_idx+1, i)
                        ocr_text = ""
                        try:
                            ocr_text = pytesseract.image_to_string(pil_img, lang="hin+eng").strip()
                        except Exception:
                            ocr_text = ""
                        docs.append({"type":"image","file":stem,"page":page_idx+1,"image_path":img_path,"ocr_text":ocr_text,"name":os.path.basename(img_path)})
                except Exception:
                    log.exception("image extraction sub-failure")
    except Exception:
        log.exception("extract_images_and_ocr failed")
    return docs

def extract_text_chunks(pdf_path: str) -> List[Dict]:
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        total = len(doc)
        num_pages = total if MAX_PAGES is None else min(total, MAX_PAGES)
        for i in range(num_pages):
            page = doc[i]
            text = page.get_text("text") or ""
            if len(text.strip()) < 30:
                # OCR fallback for scanned pages
                pix = page.get_pixmap()
                with Image.open(BytesIO(pix.tobytes("png"))) as im:
                    try:
                        text = (text + "\n" + pytesseract.image_to_string(im, lang="hin+eng")).strip()
                    except Exception:
                        pass
            for c in chunk_text(text):
                chunks.append({"type":"text","file":stem,"page":i+1,"text":c})
    except Exception:
        log.exception("extract_text_chunks failed")
    return chunks

def build_or_load_faiss(embs: np.ndarray):
    global faiss_index, faiss_dim
    if faiss_index is None:
        faiss_dim = embs.shape[1]
        # HNSW Flat
        index = faiss.IndexHNSWFlat(faiss_dim, 32)
        index.hnsw.efConstruction = 200
        index.add(embs)
        faiss_index = index
    else:
        try:
            faiss_index.add(embs)
        except Exception:
            log.exception("faiss add failed")

def persist_to_mongo_and_faiss(items: List[Dict], embed_model: SentenceTransformer):
    if not items:
        return
    texts = []
    for it in items:
        if it.get("type") == "text":
            texts.append(it["text"])
        elif it.get("type") == "table":
            texts.append(json.dumps(it.get("table_data", it)))
        elif it.get("type") == "image":
            texts.append(it.get("ocr_text") or it.get("name") or "")
        else:
            texts.append(str(it))
    # compute embeddings
    try:
        embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True).astype("float32")
    except Exception:
        log.exception("Embedding failed")
        return
    # prepare docs
    docs = []
    for it, e, txt in zip(items, embs, texts):
        doc = dict(it)  # copy
        doc["embedding"] = e.tolist()
        doc["text_for_search"] = txt
        docs.append(doc)
    # insert
    try:
        collection.insert_many(docs, ordered=False)
    except Exception:
        log.exception("Mongo insert_many failed")
    # save to faiss
    try:
        build_or_load_faiss(embs)
        # write local faiss file
        faiss.write_index(faiss_index, FAISS_PATH)
    except Exception:
        log.exception("faiss operations failed")

def ingest_pdf_file(pdf_path: str, embed_model: SentenceTransformer):
    log.info("Ingesting %s", pdf_path)
    text_chunks = extract_text_chunks(pdf_path)
    image_docs = extract_images_and_ocr(pdf_path)
    table_docs = []
    try:
        tables = extract_tables(pdf_path)
        for t in tables:
            table_docs.append({"type":"table","file":os.path.splitext(os.path.basename(pdf_path))[0],"table_data":t})
    except Exception:
        log.debug("no tables or extraction failed")
    all_items = text_chunks + table_docs + image_docs
    persist_to_mongo_and_faiss(all_items, embed_model)
    log.info("Done ingest %s (text=%d tables=%d images=%d)", os.path.basename(pdf_path), len(text_chunks), len(table_docs), len(image_docs))

def main():
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        log.exception("Failed to load embed model")
        return
    pdfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        log.warning("No PDFs in %s", PDF_DIR)
        return
    # parallel ingestion
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(ingest_pdf_file, p, embed_model) for p in pdfs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                log.exception("ingest thread failed")
    log.info("All done")

if __name__ == "__main__":
    main()
