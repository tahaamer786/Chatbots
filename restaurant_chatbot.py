from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import time
from threading import Thread

# Optional imports
try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    Pinecone = None
    OpenAI = None

# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()  # .env file in root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("Missing environment variables!")

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==========================
# GLOBAL CLIENTS
# ==========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
index = None
index_ready = False

# ==========================
# BACKGROUND PINECONE INIT
# ==========================
def init_pinecone_index():
    global index, index_ready
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Wait max 25s for index ready
    start = time.time()
    while True:
        indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX in indexes or time.time() - start > 25:
            break
        time.sleep(1)

    index = pc.Index(PINECONE_INDEX)
    index_ready = True
    print("✅ Pinecone index ready!")

@app.on_event("startup")
def startup_event():
    Thread(target=init_pinecone_index).start()  # Non-blocking

# ==========================
# REQUEST MODELS
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

@app.post("/upload-data")
def upload_data(request: UploadRequest):
    if not index_ready:
        raise HTTPException(
            status_code=503,
            detail="Pinecone index not ready. Try again in a few seconds."
        )

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
    if not index_ready:
        raise HTTPException(
            status_code=503,
            detail="Pinecone index not ready. Try again in a few seconds."
        )

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    # 1️⃣ Get embedding
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small", input=request.query
    ).data[0].embedding

    # 2️⃣ Search Pinecone
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    context = "\n".join([m["metadata"]["text"] for m in results["matches"]]) if results["matches"] else ""

    # 3️⃣ Prompt for OpenAI
    prompt = f"""
You are a restaurant AI assistant.
Answer ONLY using the context below.
If answer is not found, say "Information not available."

Context:
{context}

Question:
{request.query}
"""

    # 4️⃣ Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful restaurant assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content
    return {"answer": answer}
