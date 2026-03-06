from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from app.models.schemas import QueryRequest
from app.services.llm_service import generate_answer
from app.services.embedding_service import create_or_load_index, supported_file_types
import os
import shutil
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from app.services.retrieval_service import initialize_bm25

limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)
router = APIRouter()

# ----------------- Health check -----------------
@router.get("/")
def health():
    return {"status": "RAG Assistant Running"}

# ----------------- Ask question endpoint with rate limiting -----------------
@router.post("/ask")
@limiter.limit("5/minute")  # <- decorate the endpoint
async def ask_question(request: Request, payload: QueryRequest):
    index = request.app.state.index
    stored_docs = request.app.state.stored_docs

    if not index or not stored_docs:
        return {"answer": "No documents uploaded yet. Please upload documents first."}

    answer = generate_answer(payload.question, index, stored_docs)
    return {"answer": answer}

# ----------------- Upload documents endpoint -----------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    """
    Uploads a file, validates type, saves to data dir, and reloads the FAISS index.
    """
    index = request.app.state.index
    stored_docs = request.app.state.stored_docs

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in supported_file_types():
        return {"message": f"Unsupported file type: {ext}", "total_documents": len(stored_docs)}

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded file saved: {file_path}")

    # Reload FAISS index after upload
    index, stored_docs = create_or_load_index()

    # Reinitialize BM25 after new documents
    initialize_bm25(stored_docs)

    request.app.state.index = index
    request.app.state.stored_docs = stored_docs

    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "total_documents": len(stored_docs)
    }

# ----------------- Chat UI endpoint -----------------
@router.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """
    Returns the HTML chat interface.
    """
    template_path = "templates/index.html"
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Chat template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()