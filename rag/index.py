from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

pdf_path = BASE_DIR / ("llms.pdf")

print("PDF exists:", pdf_path.exists())  # should print True

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print(docs[7])

## splittinh
## splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

#embedding - using HuggingFace (free, no API key required)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_documents(
    documents = chunks ,
    embedding = embedding_model,
    url="http://localhost:6333",
    collection_name  = "rag"

)
print("Indexing completed")


