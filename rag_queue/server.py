"""
FastAPI Server for RAG with Background Tasks

Endpoints:
- POST /chat: Process a message asynchronously
- GET /status/{job_id}: Check job status
"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

app = FastAPI(title="RAG API")

# In-memory store for results
results_store: Dict[str, dict] = {}

# Initialize HuggingFace client
hf_client = InferenceClient(
    model="Qwen/Qwen2.5-72B-Instruct",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag",
    embedding=embedding_model
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    job_id: str
    status: str


def process_query(job_id: str, query: str):
    """Process a query in the background."""
    try:
        # Mark as processing
        results_store[job_id] = {
            "job_id": job_id,
            "query": query,
            "status": "processing"
        }
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response using HuggingFace chat completion
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        
        # Store result
        results_store[job_id] = {
            "job_id": job_id,
            "query": query,
            "response": response_text,
            "status": "completed"
        }
        
    except Exception as e:
        results_store[job_id] = {
            "job_id": job_id,
            "query": query,
            "error": str(e),
            "status": "failed"
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatMessage, background_tasks: BackgroundTasks):
    """
    Submit a message for RAG processing.
    
    Returns a job_id immediately. Use /status/{job_id} to get the result.
    """
    job_id = str(uuid.uuid4())
    
    # Add task to background
    background_tasks.add_task(process_query, job_id, payload.message)
    
    return ChatResponse(job_id=job_id, status="processing")


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status and result of a job."""
    if job_id in results_store:
        return results_store[job_id]
    return {"job_id": job_id, "status": "not_found"}


@app.get("/results")
async def list_results():
    """List all results."""
    return {"results": list(results_store.values())}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "RAG API",
        "endpoints": {
            "POST /chat": "Send a message for processing",
            "GET /status/{job_id}": "Check job status",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
