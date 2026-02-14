from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

# Load env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("Missing environment variables!")

# Import after env check
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production me specific domain laga dena
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
# STARTUP
# ==========================
@app.on_event("startup")
def startup_event():
    global openai_client, index, INDEX_READY

    try:
        # OpenAI init
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI initialized")

        # Pinecone init
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Get existing indexes safely (Pinecone v3)
        existing_indexes = pc.list_indexes().names()

        # Create if not exists
        if PINECONE_INDEX not in existing_indexes:
            print("⚠️ Creating Pinecone index...")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("✅ Pinecone index created")

        index = pc.Index(PINECONE_INDEX)
        INDEX_READY = True
        print("✅ Pinecone ready")

    except Exception as e:
        INDEX_READY = False
        print("❌ Startup failed:", str(e))

# ==========================
# EMBEDDING
# ==========================
def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

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
    return {
        "status": "API running",
        "pinecone_ready": INDEX_READY
    }

# ==========================
# UPLOAD DATA
# ==========================
@app.post("/upload-data")
def upload_data(request: UploadRequest):
    if not INDEX_READY:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not ready yet."
        )

    vectors = []

    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": text}
        })

    index.upsert(vectors=vectors)

    return {
        "status": "success",
        "documents_uploaded": len(vectors)
    }

# ==========================
# CHAT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    if not INDEX_READY:
        return {"answer": "Service initializing, please try again."}

    query_embedding = get_embedding(request.query)

    results = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
    )

    matches = results.get("matches", [])
    context = "\n".join(
        [m["metadata"]["text"] for m in matches if "metadata" in m]
    )

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

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful restaurant assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"answer": response.choices[0].message.content}
