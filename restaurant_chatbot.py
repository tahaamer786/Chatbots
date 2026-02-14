from fastapi import FastAPI, HTTPException, BackgroundTasks
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

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# EMBEDDING
# ==========================
def get_embedding(text: str):
    if not openai_client:
        return [0.0] * 1536
    try:
        return openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    except Exception as e:
        print("❌ Embedding failed:", e)
        return [0.0] * 1536

# ==========================
# MODELS
# ==========================
class UploadRequest(BaseModel):
    texts: list[str]

class ChatRequest(BaseModel):
    query: str

# ==========================
# BACKGROUND INIT
# ==========================
def init_clients():
    global openai_client, index, INDEX_READY

    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI initialized")
    except Exception as e:
        print("❌ OpenAI init failed:", e)

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        for attempt in range(12):
            try:
                existing_indexes = [idx["name"] for idx in pc.list_indexes()]
                if PINECONE_INDEX in existing_indexes:
                    index = pc.Index(PINECONE_INDEX)
                    INDEX_READY = True
                    print("✅ Pinecone index ready")
                    return
            except Exception as e:
                print(f"⚠️ Pinecone check failed (attempt {attempt+1}): {e}")
            time.sleep(5)
        # Create if missing
        print("⚠️ Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        index = pc.Index(PINECONE_INDEX)
        INDEX_READY = True
        print("✅ Pinecone index created and ready")
    except Exception as e:
        INDEX_READY = False
        print("❌ Pinecone init failed:", e)

@app.on_event("startup")
def startup_event():
    from threading import Thread
    # Start init in background thread (non-blocking)
    Thread(target=init_clients, daemon=True).start()
    print("⚡ Startup complete, initializing clients in background")

# ==========================
# HEALTH
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
        raise HTTPException(status_code=503, detail="Pinecone not ready yet.")
    vectors = [{
        "id": str(uuid.uuid4()),
        "values": get_embedding(text),
        "metadata": {"text": text}
    } for text in request.texts]
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        print("❌ Upsert failed:", e)
        raise HTTPException(status_code=500, detail="Failed to upload data.")
    return {"status": "success", "documents_uploaded": len(vectors)}

# ==========================
# CHAT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query required")
    if not INDEX_READY:
        return {"answer": "Service initializing, please try again."}

    query_embedding = get_embedding(request.query)

    try:
        results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
        matches = results.get("matches", [])
        context = "\n".join([m["metadata"]["text"] for m in matches if "metadata" in m])
    except Exception as e:
        print("❌ Query failed:", e)
        return {"answer": "Search error, try again later."}

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
