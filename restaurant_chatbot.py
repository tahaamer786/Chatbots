from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

# Load Env
load_dotenv()

# Railway par variables os.environ se uthana zyada safe hota hai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

try:
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI
except ImportError:
    raise RuntimeError("Required libraries (openai, pinecone-client) not found!")

app = FastAPI(title="Restaurant RAG Chatbot API")

# CORS Settings (Update with your production frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = None
index = None


@app.on_event("startup")
def startup_event():
    global openai_client, index

    print(f"Checking Keys: OpenAI: {bool(OPENAI_API_KEY)}, Pinecone: {bool(PINECONE_API_KEY)}")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("CRITICAL ERROR: Environment variables are missing!")
        return

    try:
        # Initialize OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists, else create
        existing_indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(PINECONE_INDEX)
        print("Successfully connected to OpenAI and Pinecone")
    except Exception as e:
        print(f"Startup failed: {e}")


class UploadRequest(BaseModel):
    texts: list[str]


class ChatRequest(BaseModel):
    query: str


def get_embedding(text: str):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


@app.get("/")
def health():
    return {
        "status": "API running",
        "openai_connected": bool(openai_client),
        "pinecone_connected": bool(index)
    }


@app.post("/upload-data")
def upload_data(request: UploadRequest):
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone not initialized")

    vectors = []
    for text in request.texts:
        embedding = get_embedding(text)
        vectors.append((str(uuid.uuid4()), embedding, {"text": text}))

    index.upsert(vectors)
    return {"status": "success", "uploaded_count": len(vectors)}


@app.post("/chat")
def chat(request: ChatRequest):
    global index, openai_client

    if not openai_client or not index:
        return {"answer": "Error: Backend components not initialized. Check Railway Variables."}

    try:
        # 1. Get Query Embedding
        query_embedding = get_embedding(request.query)

        # 2. Search Pinecone
        results = index.query(vector=query_embedding, top_k=2, include_metadata=True)

        # Check if we got any results
        if not results["matches"]:
            return {"answer": "Information not available. (Database is empty)"}

        context = "\n".join([m["metadata"]["text"] for m in results["matches"]])

        # 3. Generate Answer
        prompt = f"Answer using the context:\n{context}\n\nQuestion: {request.query}"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful restaurant assistant. Use ONLY provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"answer": f"Runtime Error: {str(e)}"}