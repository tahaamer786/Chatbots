from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

# Optional: Pinecone + OpenAI
try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    Pinecone = None
    OpenAI = None

# ==========================
# LOAD LOCAL ENV
# ==========================
load_dotenv()  # loads .env file in root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "dummy"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "dummy"
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "dummy-index"

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
# GLOBAL CLIENTS PLACEHOLDER
# ==========================
openai_client = None
index = None

# ==========================
# STARTUP EVENT
# ==========================
@app.on_event("startup")
def startup_event():
    global openai_client, index

    if OpenAI:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    if Pinecone:
        pc = Pinecone(api_key=PINECONE_API_KEY)
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
# EMBEDDING FUNCTION
# ==========================
def get_embedding(text: str):
    if openai_client:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    return [0.0]*1536  # dummy for local testing

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
    vectors = []

    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append((str(uuid.uuid4()), embedding, {"text": text}))

    if index:
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

    if not openai_client:
        return {"answer": f"Dummy answer for: {request.query}"}

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
