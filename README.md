# Token 🤖

A collection of LLM experiments including prompting techniques, RAG systems, and AI agents.

## 📋 Overview

| Module | Description |
|--------|-------------|
| `prompts/` | Zero-shot, Chain of Thought, Few-shot prompting |
| `rag/` | RAG system with PDF indexing using Qdrant |
| `rag_queue/` | Async RAG API with HuggingFace + FastAPI |
| `lang_graph/` | LangGraph with conditional edges & smart routing |
| `weather_agent/` | AI agent with tool calling |
| `ollama-fastapi/` | Local LLM API server |

## 🗂️ Project Structure

```
tokenise/
├── prompts/
│   ├── zero.py          # Zero-shot prompting
│   ├── cot.py           # Chain of Thought with chat
│   └── few.py           # Few-shot prompting
├── rag/
│   ├── index.py         # PDF indexing to Qdrant
│   └── chat.py          # RAG chat interface
├── rag_queue/
│   ├── server.py        # FastAPI server with background tasks
│   ├── docker-compose.yml
│   └── requirements.txt
├── lang_graph/
│   └── chat.py          # Conditional edges & smart routing
├── weather_agent/
│   ├── agent.py         # AI agent with tools
│   └── main.py
└── ollama-fastapi/
    └── server.py        # Ollama API server
```

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/vedsub/Token.git
cd Token
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### 2. RAG Queue API (HuggingFace)

```bash
# Start services
cd rag_queue
docker-compose up -d

# Index your PDF
cd ../rag
python index.py

# Start API
cd ../rag_queue
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /chat` - Submit a query (returns job_id)
- `GET /status/{job_id}` - Get result
- `GET /docs` - Swagger UI

### 3. Prompting Examples

```bash
python prompts/cot.py   # Interactive CoT chat
python prompts/zero.py  # Zero-shot example
```

### 4. LangGraph (Conditional Routing)

```bash
cd lang_graph
python chat.py
```

**Routing Logic:**
- Messages with "help" or "?" → Help Node
- Messages with "joke" → Joke Node
- Default → Chatbot Node

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM (Cloud) | HuggingFace (Qwen2.5-72B), Google Gemini |
| LLM (Local) | Ollama (Gemma 3) |
| Graph Framework | LangGraph |
| Vector DB | Qdrant |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| API | FastAPI |
| Queue | Valkey (Redis-compatible) |

## 📝 Environment Variables

```env
HUGGINGFACE_TOKEN=your_hf_token
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

## 📄 License

MIT License
