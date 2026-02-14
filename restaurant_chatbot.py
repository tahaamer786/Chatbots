from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import time

# Load env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise Exception("Missing environment variables!")

# Import after env
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# --------------------------
# FastAPI init
# --------------------------
app = FastAPI(title="Restaurant RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me domain daal dena
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Globals
# --------------------------
openai_client = None
index = None
INDEX_READY = False

# --------------------------
# Initialize clients (lazy-safe)
# --------------------------
def init_clients():
    global openai_client, index, INDEX_READY
    if openai_client is None:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("✅ OpenAI initialized")
        except Exception as e:
            print("❌ OpenAI init failed:", str(e))
            openai_client = None

    if index is None:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)

            # Retry to check index
            attempts = 5
            for i in range(attempts):
                existing_indexes = [idx["name"] for idx in pc.list_indexes()]
                if PINECONE_INDEX in existing_indexes:
                    break
                print(f"⚠️ Pinecone index not ready, retrying... ({i+1}/{attempts})")
                time.sleep(3)
            else:
                print("⚠️ Pinecone index missing, creating...")
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )

            index = pc.Index(PINECONE_INDEX)
            INDEX_READY = True
            print("✅ Pinecone ready")
        except Exception as e:
            print("❌ Pinecone init failed:", str(e))
            index = None
            INDEX_READY = False

# --------------------------
# Embedding function
# --------------------------
def get_embedding(text: str):
    init_clients()  # Ensure clients ready
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI not initialized yet.")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# --------------------------
# Models
# --------------------------
class UploadRequest(BaseModel):
    texts: list[str]

class ChatRequest(BaseModel):
    query: str

# --------------------------
# Health check
# --------------------------
@app.get("/")
def health():
    return {"status": "API running", "pinecone_ready": INDEX_READY}

# --------------------------
# Upload data
# --------------------------
@app.post("/upload-data")
def upload_data(request: UploadRequest):
    init_clients()
    if not INDEX_READY:
        raise HTTPException(status_code=503, detail="Pinecone not ready yet.")

    vectors = []
    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": text}
        })

    index.upsert(vectors=vectors)
    return {"status": "success", "documents_uploaded": len(vectors)}

# --------------------------
# Chat endpoint
# --------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    init_clients()
    if not INDEX_READY:
        return {"answer": "Service initializing, please try again in a few seconds."}

    query_embedding = get_embedding(request.query)
    results = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
    )

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
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful restaurant assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"answer": response.choices[0].message.content}
