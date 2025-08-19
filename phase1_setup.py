import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # For OpenAI embeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Alternative for HuggingFace

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# llm = ChatOpenAI(
#     openai_api_key=openrouter_key,
#     openai_api_base="https://openrouter.ai/api/v1",  
#     model="mistralai/mistral-7b-instruct:free",
   
# )
from langchain_community.vectorstores import FAISS



# ✅ Step 1: Load raw PDF(s)

loader = TextLoader("data/medical.txt", encoding="utf-8")
documents = loader.load()

# ✅ Step 2: Create Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# ✅ Step 3: Create Vector Embeddings
# embeddings = OpenAIEmbeddings(OPENROUTER_API_KEY=OPENROUTER_API_KEY)
# embeddings = HuggingFaceEmbeddings(model_name="google/flan-t5-large")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# If using HuggingFace instead:
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Step 4: Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ FAISS Vector Store created and saved.")
