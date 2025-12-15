# Token 🤖

A collection of LLM prompting techniques and API integrations demonstrating various approaches to working with AI models including Google Gemini and Ollama.

## 📋 Overview

This project showcases different prompting strategies and LLM integration patterns:

- **Zero-Shot Prompting** - Direct prompting without examples
- **Chain of Thought (CoT) Prompting** - Step-by-step reasoning approach
- **Few-Shot Prompting** - Learning from examples
- **Local LLM API Server** - FastAPI server using Ollama with Gemma 3

## 🗂️ Project Structure

```
tokenise/
├── prompts/
│   ├── zero.py        # Zero-shot prompting example
│   ├── cot.py         # Chain of Thought with interactive chat
│   └── few.py         # Few-shot prompting (WIP)
├── ollama-fastapi/
│   ├── server.py      # FastAPI server for Ollama
│   └── requirements.txt
├── main.py            # Basic Gemini API usage
├── gemini.py          # Gemini via OpenAI-compatible API
├── requirements.txt
└── .env               # API keys (not tracked)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally (for local LLM features)
- Google Gemini API key (for cloud features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vedsub/Token.git
   cd Token
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Running Ollama (for local LLM)

1. **Start Ollama server**
   ```bash
   ollama serve
   ```

2. **Pull the Gemma 3 model**
   ```bash
   ollama pull gemma3:270m
   ```

## 💡 Usage

### Chain of Thought Interactive Chat

Run an interactive chat session with CoT reasoning:

```bash
python prompts/cot.py
```

The assistant will break down problems using the **Start → Plan → Action → Output** format.

### Zero-Shot Prompting

```bash
python prompts/zero.py
```

### Ollama FastAPI Server

Start the local API server:

```bash
cd ollama-fastapi
uvicorn server:app --reload
```

**Endpoints:**
- `GET /` - Health check
- `POST /chat` - Send a message to Gemma 3

**Example request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '"Hello, how are you?"'
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Cloud LLM | Google Gemini 2.5 Flash |
| Local LLM | Ollama with Gemma 3 (270M) |
| API Framework | FastAPI |
| Python SDK | OpenAI, Google GenAI |

## 📚 Prompting Techniques

### Zero-Shot Prompting
Direct queries without examples. Good for straightforward tasks where the model's pre-training is sufficient.

### Chain of Thought (CoT)
Encourages step-by-step reasoning:
1. **Start** - Understand the problem
2. **Plan** - Outline the approach
3. **Action** - Execute the plan
4. **Output** - Provide the final answer

### Few-Shot Prompting
Provides examples to guide the model's responses (coming soon).

## 📝 Environment Variables

Create a `.env` file with:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
