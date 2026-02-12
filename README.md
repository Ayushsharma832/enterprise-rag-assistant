# Enterprise RAG Conversational Assistant

A production-style Retrieval-Augmented Generation (RAG) system built using FastAPI, FAISS, and Groq LLM.

## Features
- Document ingestion and chunking
- Sentence-transformer embeddings
- FAISS vector database
- Semantic retrieval
- Context-grounded LLM responses
- REST API using FastAPI
- Swagger testing interface

## Tech Stack
- Python
- FastAPI
- FAISS
- Sentence Transformers
- Groq LLM (Llama 3.1)
- Uvicorn

## Run Locally

1. Create virtual environment
2. Install dependencies
3. Add GROQ_API_KEY in `.env`
4. Run:
   uvicorn app.main:app --reload

API Docs:
http://127.0.0.1:8000/docs
