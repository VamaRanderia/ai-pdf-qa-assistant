# AI PDF Q&A Assistant 📄🤖

An AI-powered document interaction tool that enables users to upload PDF documents and ask questions in natural language. Powered by Retrieval-Augmented Generation (RAG) and Large Language Models.

---

## 🌟 Features

- **Document Ingestion**: Seamless PDF upload and text extraction.
- **Intelligent Chunking & Embeddings**: Splits documents into semantically coherent passages.
- **Vector Search**: Fast similarity search using vector embeddings.
- **Context-Aware Answers**: Accurate, cited answers based on the uploaded documents.
- **Interactive UI**: Simple, responsive interface for uploading PDFs and chatting.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment tool (`venv` or `conda`)
- An API Key (e.g., Google Gemini, OpenAI, or Anthropic)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VamaRanderia/ai-pdf-qa-assistant.git
   cd ai-pdf-qa-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```text
├── app.py              # Main application entry point (UI & interaction)
├── rag/                # RAG pipeline modules (chunking, embedding, retrieval)
├── requirements.txt    # Project dependencies
├── .env.example        # Template for environment variables
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## 📝 License

This project is licensed under the MIT License.
