from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ==========================
# LOAD ENV (Local ke liye)
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

if not PINECONE_INDEX:
    raise RuntimeError("PINECONE_INDEX not set")

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
    allow_origins=["*"],  # demo ke liye open (production me restrict karein)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# INIT CLIENTS
# ==========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX)

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
    return {"status": "API running"}

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
# UPLOAD DATA
# ==========================
@app.post("/upload-data")
def upload_data(request: UploadRequest):
    vectors = []

    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append((str(uuid.uuid4()), embedding, {"text": text}))

    index.upsert(vectors)

    return {
        "status": "success",
        "documents_uploaded": len(vectors)
    }

# ==========================
# CHAT ENDPOINT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    query_embedding = get_embedding(request.query)

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results.get("matches", [])

    if not matches:
        return {"answer": "Information not available."}

    context = "\n".join(
        [m["metadata"]["text"] for m in matches]
    )

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
