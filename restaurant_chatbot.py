from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ==========================
# ENV SETUP
# ==========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not PINECONE_INDEX or not OPENAI_API_KEY:
    raise Exception("Missing environment variables")

# ==========================
# FASTAPI INIT
# ==========================
app = FastAPI(title="Restaurant RAG Chatbot API")

# ==========================
# CORS MIDDLEWARE
# ==========================
origins = [
    "http://localhost:3000",               # Local frontend testing
    "https://your-frontend-url.vercel.app" # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# OPENAI CLIENT
# ==========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# PINECONE INIT
# ==========================
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [index["name"] for index in pc.list_indexes()]

if PINECONE_INDEX not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(PINECONE_INDEX)

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
    try:
        vectors = []

        for text in request.texts:
            embedding = get_embedding(text)
            vectors.append((
                str(uuid.uuid4()),
                embedding,
                {"text": text}
            ))

        index.upsert(vectors)

        return {
            "status": "success",
            "documents_uploaded": len(vectors)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================
# CHAT ENDPOINT
# ==========================
@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # 1️⃣ Get query embedding
        query_embedding = get_embedding(request.query)

        # 2️⃣ Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=2,
            include_metadata=True
        )

        if not results["matches"]:
            return {"answer": "No relevant information found."}

        # 3️⃣ Collect context
        context = "\n".join(
            [match["metadata"]["text"] for match in results["matches"]]
        )

        # 4️⃣ Prompt
        prompt = f"""
You are a restaurant AI assistant.
Answer ONLY using the context below.
If answer is not found, say "Information not available."

Context:
{context}

Question:
{request.query}
"""

        # 5️⃣ Generate Response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful restaurant assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
