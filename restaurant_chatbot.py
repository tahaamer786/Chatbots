from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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
# ENV SETUP
# ==========================
# Local development ke liye .env load
if os.environ.get("RAILWAY_ENV") != "true":
    from dotenv import load_dotenv
    load_dotenv()  # Only local

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

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
        # OpenAI client
        if OpenAI and OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Pinecone client
        if Pinecone and PINECONE_API_KEY:
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

        print("✅ Startup complete: OpenAI & Pinecone initialized")
    except Exception as e:
        print("❌ Startup failed:", e)

# ==========================
# EMBEDDING FUNCTION
# ==========================
def get_embedding(text: str):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
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
    global index
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

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
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    # Query embedding
    query_embedding = get_embedding(request.query)

    # Pinecone search
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    context = "\n".join([m["metadata"]["text"] for m in results["matches"]]) if results["matches"] else ""

    # Prompt
    prompt = f"""
You are a restaurant AI assistant.
Answer ONLY using the context below.
If answer is not found, say "Information not available."

Context:
{context}

Question:
{request.query}
"""

    # OpenAI response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful restaurant assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content
    return {"answer": answer}
