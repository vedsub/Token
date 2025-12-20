"""
FastAPI Server for RAG Queue System

Endpoints:
- POST /chat: Enqueue a message for processing
- POST /result: Receive processed results from worker
- GET /status/{job_id}: Check job status
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio

app = FastAPI(title="RAG Queue API")

# In-memory store for results (use Redis/DB in production)
results_store: Dict[str, dict] = {}


class ChatMessage(BaseModel):
    """Request model for /chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    job_id: str
    status: str


class ResultPayload(BaseModel):
    """Request model for /result endpoint (from worker)."""
    job_id: str
    query: str
    response: Optional[str] = None
    error: Optional[str] = None
    status: str


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatMessage):
    """
    Enqueue a message for RAG processing.
    
    The message is added to the queue and processed asynchronously.
    Use /status/{job_id} to check the result.
    """
    from client.rq_client import rq_client
    
    job_id = rq_client.enqueue_query(payload.message)
    
    return ChatResponse(
        job_id=job_id,
        status="queued"
    )


@app.post("/result")
async def receive_result(payload: ResultPayload):
    """
    Receive processed results from the worker.
    
    This endpoint is called by the worker after processing a query.
    """
    results_store[payload.job_id] = {
        "job_id": payload.job_id,
        "query": payload.query,
        "response": payload.response,
        "error": payload.error,
        "status": payload.status
    }
    
    return {"status": "received", "job_id": payload.job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get the status of a job.
    
    Returns the result if completed, or current status if still processing.
    """
    # First check if result is in our store
    if job_id in results_store:
        return results_store[job_id]
    
    # Otherwise check job queue status
    from client.rq_client import rq_client
    
    job_status = rq_client.get_job_status(job_id)
    return job_status


@app.get("/results")
async def list_results():
    """List all completed results."""
    return {"results": list(results_store.values())}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "RAG Queue API",
        "endpoints": {
            "POST /chat": "Send a message for processing",
            "GET /status/{job_id}": "Check job status",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
