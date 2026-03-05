from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from app.rag import generate_answer
import os, shutil
from app.embeddings import create_or_load_index
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Enable propagation for all loggers
logging.getLogger("uvicorn").propagate = True
logging.getLogger("uvicorn.error").propagate = True
logging.getLogger("uvicorn.access").propagate = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, stored_docs
    index, stored_docs = create_or_load_index()
    yield  # App runs here


app = FastAPI(
    title="Enterprise RAG Assistant",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    question: str


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: PlainTextResponse("Rate limit exceeded", status_code=429))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

@app.get("/")
def health():
    return {"status": "RAG Assistant Running"}


@app.post("/ask")
@limiter.limit("5/minute")
def ask_question(request: Request, payload: QueryRequest):
    answer = generate_answer(payload.question, index, stored_docs)
    return {"answer": answer}


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    global index, stored_docs  # must be first

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf", ".docx"]:
        return {"message": f"Unsupported file type: {ext}", "total_documents": len(stored_docs)}

    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Rebuild or incrementally update index
    index, stored_docs = create_or_load_index()

    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "total_documents": len(stored_docs)
    }

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    with open("templates/index.html", "r") as f:
        return f.read()