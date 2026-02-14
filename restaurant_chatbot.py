from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time

# Optional Pinecone + OpenAI
try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    Pinecone = None
    OpenAI = None

# ==========================
# ENV SETUP
# ==========================
if os.environ.get("RAILWAY_ENV") != "true":
    from dotenv import load_dotenv
    load_dotenv()  # Local only

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("Missing environment variables! Check .env or Railway dashboard.")

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

origins = [
    "http://localhost:3000",
    "https://your-frontend-url.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# GLOBAL CLIENTS
# ==========================
openai_client = None
index = None

# ==========================
# STARTUP EVENT
# ==========================
@app.on_event("startup")
def startup_event():
    global openai_client, index
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [i["name"] for i in pc.list_indexes()]

        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # Wait until index is ready (max 25s)
        timeout = 25
        start = time.time()
        while True:
            indexes = [i["name"] for i in pc.list_indexes()]
            if PINECONE_INDEX in indexes or time.time() - start > timeout:
                break
            time.sleep(1)

        index = pc.Index(PINECONE_INDEX)
        print("✅ OpenAI & Pinecone initialized, backend ready")

    except Exception as e:
        print("❌ Startup failed:", e)
        raise e

# ==========================
# EMBEDDING FUNCTION
# ==========================
def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ==========================
# REQUEST MODELS
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
    return {"status": "API running"}

# ==========================
# UPLOAD DATA
# ==========================
@app.post("/upload-data")
def upload_data(request: UploadRequest):
    if not index:
        raise HTTPException(
            status_code=500,
            detail="Pinecone index still initializing, wait a few seconds and retry."
        )

    vectors = []
    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append((str(uuid.uuid4()), embedding, {"text": text}))

    index.upsert(vectors)
    return {"status": "success", "documents_uploaded": len(vectors)}

# ==========================
# CHAT ENDPOINT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    if not index:
        raise HTTPException(
            status_code=500,
            detail="Pinecone index still initializing, wait a few seconds and retry."
        )

    query_embedding = get_embedding(request.query)
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    context = "\n".join([m["metadata"]["text"] for m in results["matches"]]) if results["matches"] else ""

    prompt = f"""
You are a restaurant AI assistant.
Answer ONLY using the context below.
If answer is not found, say "Information not available."

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
    answer = response.choices[0].message.content
    return {"answer": answer}
