# GPT Integration

Features:
- Gemini API (Hindi/English)
- MongoDB Atlas (vector search) + index creation script
- PDF ingestion: text, tables (Camelot), images (PyMuPDF)
- OCR (pytesseract) for images - supports hin+eng
- Embeddings (sentence-transformers) + FAISS local index for speed
- FastAPI endpoints for ask, ingest, TTS, optional STT
- Flutter sample client and shared preferences cache for last 2 chats
- Tests (pytest) and CI via GitHub Actions

## Quick start
1. Copy .env.example → .env and fill GEMINI_API_KEY and MONGODB_URI
2. Create Python venv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
