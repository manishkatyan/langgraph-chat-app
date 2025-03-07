# Multi-Agentic Assistant with Advanced RAG Search

A sophisticated chat application that combines multiple AI agents for flight search, news retrieval, and company financial document analysis. Built with LangGraph, Streamlit, and leveraging advanced RAG (Retrieval Augmented Generation) techniques.

## 🌟 Key Features

### 1. Multi-Agent System

- **Flight Search**: Real-time flight information using Amadeus API
- **News Search**: Current events and news via Tavily Search API
- **Company Financials**: Advanced RAG-based search through company documents

### 2. Advanced RAG Implementation

The system implements a sophisticated multi-stage retrieval pipeline:

#### Vector Store Architecture

- Uses Chroma as the vector database
- Implements persistent storage for embeddings
- Automatic PDF processing and chunking
- Document metadata preservation for source tracking

#### Hybrid Search Strategy

1. **Ensemble Retrieval**:
   - BM25 (Keyword-based search) - 20%
   - Vector Similarity Search - 40%
   - MMR (Maximal Marginal Relevance) - 40%
2. **Contextual Compression**:
   - Cohere re-ranking for result refinement
   - Top-N filtering for most relevant results
   - Citation preservation and source tracking

## 🛠 Setup

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- Required API Keys:
  - OpenAI API
  - Tavily Search API
  - Cohere API
  - Amadeus API

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   OPENAI_API_KEY=your_key
   TAVILY_API_KEY=your_key
   CO_API_KEY=your_key
   AMADEUS_CLIENT_ID=your_id
   AMADEUS_CLIENT_SECRET=your_secret
   ```

### Directory Structure

├── company_pdfs/ # Place your company PDFs here
├── chroma_db/ # Persistent vector store
├── tools/
│ ├── company_search.py
│ ├── flight_search.py
│ └── tavily_search.py
└── chat.py # Main application

## 🚀 Usage

1. Start the application:

   ```bash
   poetry run streamlit run chat.py
   ```

2. Example queries:
   - Flight Search: "Show me flights from Bangalore to Tokyo"
   - News Search: "What's happening in AI technology today?"
   - Company Search: "What was Rakuten's revenue in Q3 2023?"

## 🔍 RAG Implementation Details

### Document Processing

- Recursive character text splitting with 1000-character chunks
- 200-character overlap between chunks
- Metadata preservation for source tracking

### Retrieval Pipeline

1. Initial retrieval using ensemble method:

   - BM25 for keyword matching
   - Dense vector similarity search
   - MMR for diversity in results

2. Re-ranking:
   - Cohere's rerank-english-v3.0 model
   - Context-aware compression
   - Top-5 most relevant results

### Citation System

- Automatic inline citations [1], [2], etc.
- Source metadata tracking (file name, page number)
- Expandable source references in UI

## 📝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License

MIT License

# Ensemble Retrieval Components:

## BM25 (Okapi BM25)

A probabilistic ranking function that uses term frequency-inverse document frequency (TF-IDF) with length normalization.
Unlike basic TF-IDF, BM25 incorporates document length normalization and has tunable parameters (k1 and b) to control term frequency saturation and document length normalization.

## Vector Similarity Search

Documents and queries are converted to dense vector embeddings (typically 768-1536 dimensions) using language models like OpenAI's text-embedding-ada-002.
Similarity is computed using cosine similarity or dot product between query and document vectors, enabling semantic matching beyond exact keyword matches.

## Maximal Marginal Relevance (MMR)

An algorithm that balances relevance with diversity by selecting documents that maximize marginal relevance.
Uses the formula: MMR = λ sim(Di,Q) - (1-λ) max(sim(Di,Dj)) where λ balances between relevance to query and diversity from already selected documents.

## Contextual Compression Components:

### Cohere Re-ranking

A cross-encoder model that directly compares query-document pairs to compute relevance scores.
More computationally expensive but more accurate than bi-encoders, as it can capture complex query-document interactions.

### Top-N Filtering

A post-processing step that selects the N most relevant documents after re-ranking.
Helps maintain a manageable context window for the LLM while ensuring only the most pertinent information is included.

### Citation Preservation

A metadata tracking system that maintains document source information (filename, page numbers) throughout the retrieval pipeline.
Enables the system to generate inline citations [1], [2], etc., and provide source attribution for all retrieved information.
