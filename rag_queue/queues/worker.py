"""
Queue Worker for RAG Query Processing

This worker processes messages from the queue and calls the /result endpoint
with the generated response.
"""
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Qdrant vector store
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag",
    embedding=embedding_model
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# FastAPI server URL for callback
FASTAPI_SERVER_URL = os.getenv("FASTAPI_SERVER_URL", "http://localhost:8000")


def process_query(job_id: str, query: str) -> dict:
    """
    Process a query from the queue.
    
    This function:
    1. Retrieves relevant documents from the vector store
    2. Generates a response using Gemini LLM
    3. Calls the /result endpoint with the response
    
    Args:
        job_id: Unique identifier for this job
        query: The user's question/message
        
    Returns:
        dict: The result containing job_id and response
    """
    try:
        # Step 1: Retrieve relevant documents
        docs = retriever.invoke(query)
        
        # Step 2: Build context from retrieved documents
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Step 3: Create prompt with context
        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Step 4: Generate response using Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        result = {
            "job_id": job_id,
            "query": query,
            "response": response.text,
            "status": "completed"
        }
        
        # Step 5: Call the /result endpoint with the response
        try:
            requests.post(
                f"{FASTAPI_SERVER_URL}/result",
                json=result,
                timeout=10
            )
        except requests.RequestException as e:
            print(f"Failed to call /result endpoint: {e}")
        
        return result
        
    except Exception as e:
        error_result = {
            "job_id": job_id,
            "query": query,
            "error": str(e),
            "status": "failed"
        }
        
        # Notify failure to /result endpoint
        try:
            requests.post(
                f"{FASTAPI_SERVER_URL}/result",
                json=error_result,
                timeout=10
            )
        except requests.RequestException:
            pass
            
        return error_result