from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.services.embedding_service import create_or_load_index

# ----------------- Rate limiter -----------------
limiter = Limiter(key_func=get_remote_address)

# ----------------- Load FAISS embeddings/index -----------------
index, stored_docs = create_or_load_index()

# ----------------- FastAPI app -----------------
app = FastAPI(title="Enterprise RAG Assistant")

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Include routes
app.include_router(router, prefix="")

# Attach global state
app.state.index = index
app.state.stored_docs = stored_docs
app.state.limiter = limiter

# ----------------- Global exception handler for rate limit -----------------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return PlainTextResponse("Rate limit exceeded", status_code=429)