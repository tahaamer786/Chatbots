from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import time

# Optional: Pinecone + OpenAI
try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    Pinecone = None
    OpenAI = None

# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()  # Load .env file in root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("❌ Missing environment variables! Check .env or Railway settings")

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

origins = [
    "http://localhost:3000",  # Local frontend
    "https://your-frontend-url.vercel.app"  # Production frontend
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
# HELPER: Wait for Pinecone Index
# ==========================
def wait_for_index(pc, index_name, max_attempts=15, sleep_sec=2):
    for attempt in range(1, max_attempts + 1):
        existing = [i["name"] for i in pc.list_indexes()]
        if index_name in existing:
            print(f"✅ Pinecone index ready after {attempt} attempts")
            return True
        print(f"⏳ Waiting for Pinecone index... Attempt {attempt}")
        time.sleep(sleep_sec)
    raise Exception("❌ Pinecone index not ready after max attempts")

# ==========================
# STARTUP EVENT
# ==========================
@app.on_event("startup")
def startup_event():
    global openai_client, index

    # OpenAI client
    if OpenAI:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")

    # Pinecone client
    if Pinecone:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Wait until index is ready
        wait_for_index(pc, PINECONE_INDEX)
        # Create index if not exists
        existing_indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            wait_for_index(pc, PINECONE_INDEX)
        index = pc.Index(PINECONE_INDEX)
        print("✅ Pinecone client and index ready")

# ==========================
# EMBEDDING FUNCTION
# ==========================
def get_embedding(text: str):
    if openai_client:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    # fallback if OpenAI not initialized
    return [0.0]*1536

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
    global index
    if not index:
        raise HTTPException(status_code=503, detail="Pinecone index not ready. Try again later.")

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
    global index, openai_client

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    if not openai_client or not index:
        return {"answer": "Service initializing, please try again in a few seconds."}

    # Get embedding
    query_embedding = get_embedding(request.query)

    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)

    if not results["matches"]:
        return {"answer": "Information not available."}

    context = "\n".join([m["metadata"]["text"] for m in results["matches"]])

    # Generate response
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
