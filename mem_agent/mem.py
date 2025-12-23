import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# ================================
# Configuration
# ================================

# Qdrant Vector Store Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "mem_agent"

# ================================
# Embeddings Configuration
# ================================
# Using HuggingFace embeddings (sentence-transformers)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# LLM Configuration
# ================================
# Using HuggingFace Endpoint (Qwen2.5-72B-Instruct)
hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)
llm = ChatHuggingFace(llm=hf_llm)

# ================================
# Vector Store Configuration
# ================================
# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding
)