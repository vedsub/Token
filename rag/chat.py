from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the same embedding model used during indexing
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing Qdrant collection
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag",
    embedding=embedding_model
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 most similar chunks
)


def get_response(query: str) -> str:
    """
    Retrieve relevant context and generate a response using Gemini.
    """
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
    
    return response.text


def main():
    print("RAG Chat - Ask questions about the LLMs PDF")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nSearching and generating response...")
        response = get_response(query)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
