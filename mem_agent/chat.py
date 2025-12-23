import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document

load_dotenv()

# ================================
# Configuration
# ================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "mem_agent"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# ================================
# Initialize Components
# ================================

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM
hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)
llm = ChatHuggingFace(llm=hf_llm)

# Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL)


class MemoryAgent:
    """AI Agent with long-term memory using Qdrant vector store."""
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.conversation_history = []  # Short-term memory
        self._init_collection()
        self.vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embedding
        )
    
    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            print(f"📦 Using existing collection: {COLLECTION_NAME}")
    
    def store_memory(self, user_message: str, ai_response: str):
        """Store conversation exchange in long-term memory."""
        memory_content = f"User: {user_message}\nAssistant: {ai_response}"
        
        doc = Document(
            page_content=memory_content,
            metadata={
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "ai_response": ai_response[:500]  # Truncate for metadata
            }
        )
        
        self.vector_store.add_documents([doc])
    
    def retrieve_memories(self, query: str, k: int = 3) -> list:
        """Retrieve relevant memories based on current query."""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter={"user_id": self.user_id}
            )
            return results
        except Exception as e:
            print(f"⚠️ Memory retrieval error: {e}")
            return []
    
    def build_context(self, user_message: str) -> str:
        """Build context from long-term memories."""
        memories = self.retrieve_memories(user_message)
        
        if not memories:
            return ""
        
        context = "📚 Relevant past conversations:\n"
        for i, mem in enumerate(memories, 1):
            context += f"\n--- Memory {i} ---\n{mem.page_content}\n"
        
        return context
    
    def chat(self, user_message: str) -> str:
        """Process user message and generate response with memory."""
        
        # Retrieve relevant memories
        memory_context = self.build_context(user_message)
        
        # Build system prompt with memory
        system_prompt = """You are a helpful AI assistant with memory capabilities. 
You can remember past conversations and use them to provide more personalized responses.

If relevant memories are provided, use them to:
1. Maintain continuity in conversations
2. Remember user preferences and details
3. Provide more personalized responses

Be natural and conversational. Don't explicitly mention "memories" unless asked."""

        if memory_context:
            system_prompt += f"\n\n{memory_context}"
        
        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (short-term memory)
        for entry in self.conversation_history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = llm.invoke(messages)
        ai_response = response.content
        
        # Update short-term memory
        self.conversation_history.append({
            "user": user_message,
            "assistant": ai_response
        })
        
        # Store in long-term memory
        self.store_memory(user_message, ai_response)
        
        return ai_response
    
    def clear_conversation(self):
        """Clear short-term conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.")
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            return collection_info.points_count
        except:
            return 0


def main():
    """Interactive chat loop with memory-enabled agent."""
    print("=" * 60)
    print("🧠 Memory-Enabled AI Agent")
    print("=" * 60)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /memory - Show memory count")
    print("  /quit   - Exit the chat")
    print("=" * 60)
    
    # Get user ID for personalized memory
    user_id = input("\n👤 Enter your name (or press Enter for default): ").strip()
    if not user_id:
        user_id = "default_user"
    
    agent = MemoryAgent(user_id=user_id)
    print(f"\n✨ Hello {user_id}! I remember our past conversations.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("\n👋 Goodbye! Your memories are saved.")
                break
            elif user_input.lower() == "/clear":
                agent.clear_conversation()
                continue
            elif user_input.lower() == "/memory":
                count = agent.get_memory_count()
                print(f"📊 Total memories stored: {count}")
                continue
            
            # Get response
            print("\n🤔 Thinking...")
            response = agent.chat(user_input)
            print(f"\n🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Your memories are saved.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
