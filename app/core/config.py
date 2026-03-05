import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_FILE = os.getenv("INDEX_FILE", "faiss_index.index")
DOCS_FILE = os.getenv("DOCS_FILE", "documents.pkl")