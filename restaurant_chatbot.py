from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import time

# ==========================
# ENV
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("Missing environment variables!")

# ==========================
# IMPORT AFTER ENV
# ==========================
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# GLOBALS
# ==========================
openai_client = None
index = None
INDEX_READY = False

# ==========================
# STARTUP EVENT (CRASH-PROOF)
# ==========================
@app.on_event("startup")
def startup_event():
    global openai_client, index, INDEX_READY

    try:
        # OpenAI init
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI initialized")
    except Exception as e:
        print("❌ OpenAI init failed:", e)

    try:
        # Pinecone init
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Retry mechanism for index
        attempts = 12
        for i in range(attempts):
            try:
                existing_indexes = [idx["name"] for idx in pc.list_indexes()]
                if PINECONE_INDEX in existing_indexes:
                    index = pc.Index(PINECONE_INDEX)
                    INDEX_READY = True
                    print("✅ Pinecone index ready")
                    break
            except Exception as e:
                print(f"⚠️ Pinecone check failed (attempt {i+1}/{attempts}): {e}")
            time.sleep(5)
        else:
            print("⚠️ Pinecone index not found after retries, creating...")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)
            index = pc.Index(PINECONE_INDEX)
            INDEX_READY = True
            print("✅ Pinecone index created and ready")
    except Exception as e:
        INDEX_READY = False
        print("❌ Pinecone init failed:", e)

# ==========================
# EMBEDDING
# ==========================
def get_embedding(text: str):
    if openai_client:
        try:
            return openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
        except Exception as e:
            print("❌ Embedding failed:", e)
    return [0.0] * 1536  # fallback

# ==========================
# MODELS
# ==========================
class UploadRequest(BaseModel):
    texts: list[str]

class ChatRequest(BaseModel):
    query: str

# ==========================
# HEALTH CHECK
# ==========================
@app.get("/")
def health():
    return {"status": "API running", "pinecone_ready": INDEX_READY}

# ==========================
# UPLOAD DATA
# ==========================
@app.post("/upload-data")
def upload_data(request: UploadRequest):
    if not INDEX_READY:
        raise HTTPException(status_code=503, detail="Pinecone not ready, try again later.")

    vectors = [{
        "id": str(uuid.uuid4()),
        "values": get_embedding(text),
        "metadata": {"text": text}
    } for text in request.texts]

    index.upsert(vectors=vectors)
    return {"status": "success", "documents_uploaded": len(vectors)}

# ==========================
# CHAT ENDPOINT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query required")

    if not INDEX_READY:
        return {"answer": "Service initializing, please try again in a few seconds."}

    query_embedding = get_embedding(request.query)
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    matches = results.get("matches", [])
    context = "\n".join([m["metadata"]["text"] for m in matches if "metadata" in m])

    if not context:
        return {"answer": "Information not available."}

    prompt = f"""
You are a restaurant AI assistant.
Answer ONLY from the context below.

Context:
{context}

Question:
{request.query}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful restaurant assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        print("❌ OpenAI completion failed:", e)
        return {"answer": "AI service error, try again later."}
