# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, BM25, Sentence Transformers, and Groq Llama 3.1.

The system enables enterprise knowledge retrieval by combining semantic vector search and keyword-based retrieval to generate grounded, context-aware responses from uploaded documents.

---

## Features

### Document Ingestion

- Upload and index:
  - TXT
  - PDF
  - DOCX

- Automatic text extraction
- Chunking with overlap
- Incremental indexing of newly uploaded content

### Retrieval Pipeline

- Semantic Search using FAISS
- Keyword Search using BM25
- Hybrid Retrieval (FAISS + BM25)
- Top-K Context Retrieval

### LLM Generation

- Groq-hosted Llama 3.1
- Context-grounded prompting
- Hallucination reduction
- Enterprise-style question answering

### Reliability & Production Features

- Incremental FAISS indexing
- Duplicate chunk detection using SHA256 hashing
- Index–metadata integrity validation
- Automatic index recovery
- Structured logging
- Rate limiting using SlowAPI
- Exception handling for LLM failures

---

# Architecture

```text
                    Documents
                         |
                         v
                Text Extraction
                         |
                         v
                  Chunking Layer
                  (Overlap Enabled)
                         |
                         v
                 Sentence Embeddings
                         |
                         v
                   FAISS Index
                         |
                         |
                         v

User Query
     |
     +--------------------+
     |                    |
     v                    v
Query Embedding       BM25 Search
     |                    |
     +--------+-----------+
              |
              v
      Hybrid Retrieval
              |
              v
      Relevant Chunks
              |
              v
       Prompt Builder
              |
              v
      Groq Llama 3.1
              |
              v
      Grounded Response
```

---

# Project Structure

```text
app/
├── api/
│   └── routes.py
│
├── core/
│   ├── config.py
│   └── logging_config.py
│
├── models/
│   └── schemas.py
│
├── services/
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   └── llm_service.py
│
├── utils/
│   └── helpers.py
│
└── main.py

templates/
└── index.html

requirements.txt
README.md
```

---

# Retrieval Pipeline

## 1. Document Processing

Supported file formats:

- TXT
- PDF
- DOCX

Uploaded documents are converted into raw text.

---

## 2. Chunking

Documents are split into overlapping chunks.

Current configuration:

```python
chunk_size = 200
overlap = 50
```

Benefits:

- Improved retrieval precision
- Better context preservation
- Reduced information loss at chunk boundaries

---

## 3. Embedding Generation

Embeddings are generated using:

```text
all-MiniLM-L6-v2
```

via Sentence Transformers.

Each chunk is converted into a semantic vector representation.

---

## 4. Vector Indexing

FAISS is used for vector similarity search.

Current index type:

```text
IndexFlatL2
```

Advantages:

- Exact nearest-neighbor search
- High retrieval quality
- Simple implementation

---

## 5. Hybrid Retrieval

The system combines:

### Semantic Retrieval

FAISS vector similarity search

### Keyword Retrieval

BM25 ranking

This improves retrieval quality by capturing both:

- Semantic meaning
- Exact keyword matches

---

## 6. Grounded Generation

Retrieved chunks are injected into the prompt before LLM inference.

Prompting strategy:

```text
Answer strictly using the provided context.
If the answer is not present,
respond with:
"Information not available in the provided documents."
```

This reduces hallucinations and improves answer reliability.

---

# Incremental Indexing

Instead of rebuilding the entire vector index after every upload:

```text
Upload
→ Re-embed everything
→ Rebuild index
```

the system performs:

```text
Upload
→ Embed only new chunks
→ Append vectors
→ Update metadata
```

Benefits:

- Faster ingestion
- Reduced compute overhead
- Improved scalability

---

# Duplicate Detection

The ingestion pipeline uses SHA256 hashing.

```text
Chunk
→ SHA256 Hash
→ Duplicate Check
```

Duplicate chunks are skipped to avoid:

- Redundant embeddings
- Larger index size
- Retrieval noise

---

# Index Integrity Validation

The system validates:

```python
index.ntotal == len(stored_docs)
```

This ensures:

- Every vector has metadata
- No retrieval mismatches
- Early corruption detection

If corruption is detected:

```text
Delete corrupted index
→ Rebuild automatically
```

---

# API Endpoints

## Health Check

```http
GET /
```

Response:

```json
{
  "status": "RAG Assistant Running"
}
```

---

## Ask Question

```http
POST /ask
```

Request:

```json
{
  "question": "What is the company leave policy?"
}
```

Response:

```json
{
  "answer": "..."
}
```

Rate Limit:

```text
5 requests/minute
```

---

## Upload Document

```http
POST /upload
```

Supported formats:

- TXT
- PDF
- DOCX

Response:

```json
{
  "message": "Document uploaded successfully",
  "total_documents": 45
}
```

---

## Web Chat UI

```http
GET /chat
```

Returns a lightweight HTML interface for interacting with the assistant.

---

# Tech Stack

## Backend

- FastAPI
- Uvicorn

## Retrieval

- FAISS
- BM25

## Embeddings

- Sentence Transformers
- all-MiniLM-L6-v2

## LLM

- Groq API
- Llama 3.1 8B Instant

## NLP

- Semantic Search
- Retrieval-Augmented Generation (RAG)

---

# Future Improvements

### Retrieval

- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Metadata Filtering
- Query Expansion

### Scalability

- IVF / HNSW FAISS Indexes
- Distributed Vector Databases
- Pinecone
- Weaviate
- Qdrant

### Enterprise Features

- User Authentication
- Role-Based Access Control (RBAC)
- Audit Logging
- Multi-Tenant Support

### Performance

- Redis Caching
- Async Processing
- Streaming Responses

### Evaluation

- RAGAS Evaluation
- LangSmith Tracing
- Retrieval Benchmarking

---

# Running Locally

## Clone Repository

```bash
git clone <repository-url>
cd enterprise-rag-assistant
```

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

or

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

## Start Server

```bash
uvicorn app.main:app --reload
```

---

## API Documentation

```text
http://localhost:8000/docs
```

---

# Resume Highlights

- Built a production-style RAG architecture using FastAPI, FAISS, BM25, and Groq Llama 3.1.
- Implemented hybrid retrieval combining semantic and keyword search.
- Added chunk overlap and incremental indexing to improve retrieval quality and ingestion efficiency.
- Implemented duplicate detection, index integrity validation, and automatic recovery mechanisms.
- Protected LLM endpoints using rate limiting and exception handling.
- Designed a scalable and modular architecture for enterprise knowledge retrieval systems.
