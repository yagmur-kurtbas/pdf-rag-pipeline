from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import json


with open("groq-api.json", "r") as f:
    groq_api = json.load(f)

API_KEY = groq_api["api_key"]

def load_pdf(path: str):
    """"Loads a PDF file and returns documents."""
    loader = PyPDFLoader(path)
    return loader.load()

def split_documents(documents: list):
    """Split documents into chunks.
    Because feeding whole PDF to a LLM is huge.
    By splitting it into small parts, we aim to avoid token limit issue.
    chunk_size: split PDF into n number of characters.
    chunk_overlap: each chunk overlaps previous n number of characters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks: list):
    """Create ChromaDB vectorstore from chunks.
    Turn PDF parts into numerical vectors.
    This is called semantic search."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", # small but efficient model
    )
    return Chroma.from_documents(chunks, embeddings)

def ask_question(vectorstore, question: str) -> str:
    """Ask a question and get an answer from PDF.
    ChromaDB is a database that saves the vectors.
    It saves the embedding; not numbers or texts like normal database.
    PDF context can be anything, from any language."""
    llm = ChatGroq(
        api_key = API_KEY,
        model = "llama-3.3-70b-versatile", # big and powerful
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question) # pull the closest chunk from ChromaDB
    context = "\n".join([d.page_content for d in docs]) # merge the found chunks and feed to LLM

    prompt = f"Answer based on context:\n{context}\n\nquestion:\n{question}"
    return llm.invoke(prompt).content

if __name__ == '__main__':
    docs = load_pdf("your-file.pdf")
    print(f'Loaded {len(docs)} pages.')

    chunks = split_documents(docs)
    print(f'Split into {len(chunks)} chunks.')

    vectors = create_vectorstore(chunks)
    print(f'Vectorstore created!')

    answer = ask_question(vectors, "What is this document about?") # Adjust the prompt
    print(answer)