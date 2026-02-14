from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import time

# Optional imports
try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    Pinecone = None
    OpenAI = None

# ==========================
# LOAD ENV
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    print("⚠️ Warning: Missing env variables, using dummy values for local testing.")
    OPENAI_API_KEY = OPENAI_API_KEY or "dummy"
    PINECONE_API_KEY = PINECONE_API_KEY or "dummy"
    PINECONE_INDEX = PINECONE_INDEX or "dummy-index"

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==========================
# GLOBAL CLIENTS
# ==========================
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI else None
index = None

# ==========================
# PINECONE INITIALIZATION
# ==========================
def init_pinecone():
    global index
    if not Pinecone:
        print("⚠️ Pinecone not installed, skipping index init")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Wait until index is ready
    start = time.time()
    while True:
        indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX in indexes:
            break
        if time.time() - start > 20:
            print("⚠️ Pinecone index not ready after 20s")
            break
        time.sleep(1)

    index = pc.Index(PINECONE_INDEX)
    print("✅ Pinecone index ready")

@app.on_event("startup")
def startup_event():
    init_pinecone()  # synchronous init for simplicity

# ==========================
# MODELS
# ==========================
class UploadRequest(BaseModel):
    texts: list[str]

class ChatRequest(BaseModel):
    query: str

# ==========================
# ENDPOINTS
# ==========================
@app.get("/")
def health():
    return {"status": "API running"}

def wait_for_index(max_wait=20):
    """Wait until Pinecone index is initialized"""
    start = time.time()
    while index is None:
        if time.time() - start > max_wait:
            raise HTTPException(status_code=503, detail="Pinecone index not ready. Try again later.")
        time.sleep(1)

@app.post("/upload-data")
def upload_data(request: UploadRequest):
    wait_for_index()
    vectors = []
    for text in request.texts:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
        vectors.append((str(uuid.uuid4()), embedding, {"text": text}))
    index.upsert(vectors)
    return {"status": "success", "documents_uploaded": len(vectors)}

@app.post("/chat")
def chat(request: ChatRequest):
    wait_for_index()
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small", input=request.query
    ).data[0].embedding

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
