from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import generate_answer
from fastapi import UploadFile, File
import shutil
from app.embeddings import create_or_load_index
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI(title="Enterprise RAG Assistant")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "RAG Assistant Running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = generate_answer(request.question)
    return {"answer": answer}

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Rebuild index using all documents
    index, documents = create_or_load_index()

    return {"message": f"{file.filename} uploaded and indexed successfully", 
            "total_documents": len(documents)}

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    with open("templates/index.html", "r") as f:
        return f.read()
