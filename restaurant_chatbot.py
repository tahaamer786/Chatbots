from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

load_dotenv()

# Keys setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
openai_client = None
index = None


# Initializing function - isko har endpoint mein check karenge
def initialize_clients():
    global openai_client, index
    try:
        if openai_client is None:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY missing")
            openai_client = OpenAI(api_key=OPENAI_API_KEY)

        if index is None:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY missing")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            # Index check
            existing = [i["name"] for i in pc.list_indexes()]
            if PINECONE_INDEX not in existing:
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            index = pc.Index(PINECONE_INDEX)
    except Exception as e:
        print(f"Init Error: {e}")
        return False
    return True


class ChatRequest(BaseModel):
    query: str


class UploadRequest(BaseModel):
    texts: list[str]


@app.get("/")
def home():
    is_ready = initialize_clients()
    return {"status": "Online", "ready": is_ready}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not initialize_clients():
        raise HTTPException(status_code=500, detail="Backend keys not configured correctly on Railway")

    try:
        # Get Embedding
        embed_res = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        xq = embed_res.data[0].embedding

        # Query Pinecone
        res = index.query(vector=xq, top_k=2, include_metadata=True)
        context = "\n".join([m["metadata"]["text"] for m in res["matches"]]) if res["matches"] else ""

        if not context:
            return {"answer": "I don't have information about this. Please upload data first."}

        # Chat Completion
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a restaurant assistant. Answer strictly based on context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


@app.post("/upload-data")
async def upload_data(request: UploadRequest):
    if not initialize_clients():
        raise HTTPException(status_code=500, detail="Backend not ready")

    vectors = []
    for text in request.texts:
        embed_res = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        vectors.append((str(uuid.uuid4()), embed_res.data[0].embedding, {"text": text}))

    index.upsert(vectors)
    return {"status": "success", "count": len(vectors)}