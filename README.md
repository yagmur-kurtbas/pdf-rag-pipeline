# PDF RAG Pipeline 📄

A simple RAG (Retrieval-Augmented Generation) pipeline that answers questions about PDF documents using LangChain, ChromaDB, and Groq LLM.

## How it works
1. Load a PDF file
2. Split into chunks
3. Store chunks in ChromaDB vectorstore as embeddings
4. Ask a question → retrieve relevant chunks → LLM generates answer

## Tech Stack
- **LangChain** - RAG pipeline orchestration
- **ChromaDB** - Vector database for storing embeddings
- **HuggingFace Embeddings** - Text to vector conversion (all-MiniLM-L6-v2)
- **Groq LLM** - Fast inference with llama-3.3-70b

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pdf-rag-pipeline.git
cd pdf-rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv ragEnv
ragEnv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Groq API Key (Free)
1. Go to [groq.com](https://groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** section
4. Create a new API key
5. Add it to `groq-api.json`:
```json
{
    "api_key": "your-free-groq-api-key-here"
}
```

### 5. Add your PDF
Place your PDF file in the project directory and update the filename in `main.py`:
```python
docs = load_pdf("your-file.pdf")
```

### 6. Run
```bash
python main.py
```

## Example Output
```
Loaded 1 pages.
Split into 4 chunks.
Vectorstore created!
This document is about space exploration...
```