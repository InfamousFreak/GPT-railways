import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import camelot
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "./knowledge_pack"
DOCS_DIR = os.path.join(DATA_DIR, "documents")
IMG_DIR = os.path.join(DATA_DIR, "images")
FAISS_PATH = os.path.join(DATA_DIR, "index_hnsw.faiss")

# Path to tesseract (Windows users: update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
# ----------------------------
# Embedding model
# ----------------------------
print("🔹 Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Data holders
# ----------------------------
documents = []   # extracted text
index_to_doc = {}  # map index -> metadata

# ----------------------------
# Process PDFs
# ----------------------------
print("🔹 Processing PDFs...")

for file in os.listdir(DOCS_DIR):
    if not file.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(DOCS_DIR, file)
    print(f"📄 Extracting from {file_path} ...")

    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            # Normal text found
            documents.append(text)
            index_to_doc[len(documents) - 1] = {
                "file": file,
                "page": page_num + 1,
                "type": "text",
            }
        else:
            # Fallback to OCR
            try:
                images = convert_from_path(
                    file_path, first_page=page_num+1, last_page=page_num+1, dpi=300
                )
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        documents.append(ocr_text)
                        index_to_doc[len(documents) - 1] = {
                            "file": file,
                            "page": page_num + 1,
                            "type": "ocr",
                        }
            except Exception as e:
                print(f"⚠️ OCR failed for {file}, page {page_num+1}: {e}")

    # --- Extract tables (Camelot) ---
    try:
        tables = camelot.read_pdf(file_path, pages="all")
        for i, table in enumerate(tables):
            table_text = table.df.to_string()
            documents.append(table_text)
            index_to_doc[len(documents) - 1] = {
                "file": file,
                "page": None,
                "type": "table",
                "table_no": i,
            }
    except Exception as e:
        print(f"⚠️ Table extraction failed for {file}: {e}")

    # --- Extract images (page by page to avoid memory error) ---
    try:
        total_pages = len(reader.pages)
        for page_num in range(total_pages):
            images = convert_from_path(
                file_path, dpi=200,
                first_page=page_num+1, last_page=page_num+1
            )
            for img in images:
                img_path = os.path.join(IMG_DIR, f"{file}_page{page_num+1}.png")
                img.save(img_path, "PNG")
                documents.append(f"[IMAGE] {img_path}")
                index_to_doc[len(documents) - 1] = {
                    "file": file,
                    "page": page_num + 1,
                    "type": "image",
                    "path": img_path,
                }
    except Exception as e:
        print(f"⚠️ Image extraction failed for {file}: {e}")


print(f"✅ Extracted {len(documents)} chunks (text + tables + images + OCR)")

# ----------------------------
# Create embeddings (batched)
# ----------------------------
print("🔹 Creating embeddings...")
embeddings = []
batch_size = 32
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embeddings.extend(embs)

embeddings = np.array(embeddings)
print(f"✅ Created embeddings shape: {embeddings.shape}")

# ----------------------------
# Build FAISS index
# ----------------------------
if embeddings.shape[0] > 0:
    print("🔹 Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    index.add(embeddings)

    # Save FAISS + Pickle
    faiss.write_index(index, FAISS_PATH)
    with open(FAISS_PATH + ".pkl", "wb") as f:
        pickle.dump((index, index_to_doc), f)

    print(f"✅ Saved FAISS index at: {FAISS_PATH}")
    print(f"✅ Saved pickle mapping at: {FAISS_PATH}.pkl")
else:
    print("⚠️ No embeddings were created. Check if PDFs are empty or extraction failed.")
