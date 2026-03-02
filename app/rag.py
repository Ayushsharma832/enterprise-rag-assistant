import os
from groq import Groq
from dotenv import load_dotenv
from app.embeddings import load_all_documents, create_or_load_index, search
import logging
import time

logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# removed Load and index documents at startup and make them global in main.py to reduce loadup time during import


def generate_answer(query: str, index, stored_docs):
    if index is None or not stored_docs:
        logger.warning("No documents indexed. Retrieval unavailable.")
        return "No documents have been uploaded yet. Please upload documents first."
    start_time = time.time()

    # ---------------- RETRIEVAL TIMING ----------------
    retrieval_start = time.time()
    context_chunks = search(index, stored_docs, query)
    retrieval_time = time.time() - retrieval_start

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an enterprise AI assistant.

Answer the question strictly using ONLY the context below.
If the answer is not present in the context, respond exactly with:
"Information not available in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""
    logger.info(f"User Query: {query}")
    logger.info(f"Retrieved {len(context_chunks)} context chunks")
    logger.info(f"Retrieval Time: {retrieval_time:.3f}s")

    try:
        # ---------------- LLM TIMING ----------------
        llm_start = time.time()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        llm_time = time.time() - llm_start

        total_time = time.time() - start_time

        logger.info(f"LLM Time: {llm_time:.3f}s")
        logger.info(f"Total Request Time: {total_time:.3f}s")

        return response.choices[0].message.content
    except Exception as e:
        logger.error("LLM generation failed", exc_info=True)
        return "An internal error occurred while generating the response."
