import os, time, logging
from app.services.retrieval_service import retrieve_top_chunks
from app.core.config import GROQ_API_KEY
from groq import Groq

logger = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query: str, index, stored_docs):
    if not index or not stored_docs:
        return "No documents uploaded yet. Please upload documents first."

    context_chunks = retrieve_top_chunks(index, stored_docs, query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an enterprise AI assistant. Answer the question strictly using ONLY the context below.
If the answer is not present in the context, respond exactly with: "Information not available in the provided documents."
Context: {context}
Question: {query}
Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "An internal error occurred while generating the response."