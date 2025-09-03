import os
import base64
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS

from utils.logging_config import setup_logging
from retrieve_and_answer import answer as rag_answer
from ingest_to_mongodb import ingest_pdf_file, SentenceTransformer, EMBED_MODEL_NAME  # ingest function
# Note: ingest_to_mongodb defines ingest_pdf_file and model name

setup_logging()
log = logging.getLogger("server")
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]

app = FastAPI(title="GPT Integration Backend")

app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error":"Internal server error"})

@app.get("/")
def health():
    return {"status":"running"}

@app.post("/ask")
def ask(q: Query):
    res = rag_answer(q.question)
    return res

@app.post("/ingest/pdf")
async def ingest_pdf_upload(file: UploadFile = File(...)):
    try:
        # save file to disk
        os.makedirs("pdf", exist_ok=True)
        path = os.path.join("pdf", file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        # embed model
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        ingest_pdf_file(path, embed_model)
        return {"status":"ok","file":file.filename}
    except Exception:
        log.exception("ingest endpoint failed")
        return JSONResponse(status_code=500, content={"error":"ingest failed"})

@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):
    try:
        if lang not in {"en","hi"}:
            lang = "en"
        tts = gTTS(text=text, lang=lang)
        out = f"tts_{lang}.mp3"
        tts.save(out)
        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"audio_base64": f"data:audio/mpeg;base64,{b64}"}
    except Exception:
        log.exception("TTS failed")
        return JSONResponse(status_code=500, content={"error":"tts failed"})

# Optional STT endpoint using whisper if enabled in .env
ENABLE_STT = os.getenv("ENABLE_STT","false").lower()=="true"
if ENABLE_STT:
    try:
        import whisper
        WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")
        _whisper = whisper.load_model(WHISPER_MODEL)
    except Exception:
        log.exception("Whisper initialization failed; disabling STT")
        ENABLE_STT = False

@app.post("/stt")
async def stt(audio: UploadFile = File(...), lang_hint: str = Form("auto")):
    if not ENABLE_STT:
        return JSONResponse(status_code=400, content={"error":"STT disabled on server"})
    try:
        tmp = "_tmp_audio_input"
        with open(tmp, "wb") as f:
            f.write(await audio.read())
        result = _whisper.transcribe(tmp, language=None if lang_hint=="auto" else lang_hint)
        os.remove(tmp)
        return {"text": result.get("text","").strip()}
    except Exception:
        log.exception("STT failed")
        return JSONResponse(status_code=500, content={"error":"stt failed"})
