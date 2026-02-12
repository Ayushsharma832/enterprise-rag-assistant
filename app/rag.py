import os
from groq import Groq
from dotenv import load_dotenv
from app.embeddings import load_all_documents, create_or_load_index, search

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load and index documents at startup
index, stored_docs = create_or_load_index()

def generate_answer(query: str):
    context_chunks = search(index, stored_docs, query)

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
    print("User Query:", query)
    print("Retrieved Context:", context_chunks)


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
