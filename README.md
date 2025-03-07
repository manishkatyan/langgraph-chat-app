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

- A sophisticated keyword matching algorithm that improves upon traditional word-frequency matching. Think of it as "Ctrl+F" on steroids.
- It's smart enough to understand that if a word appears 10 times in a short document, it's probably more relevant than if it appears 10 times in a very long document. It also prevents common words from dominating the results.

## Vector Similarity Search

- Converts words and sentences into numerical representations (vectors) where similar meanings are close to each other in mathematical space.
- For example, "automobile" and "car" would be close together in this space, allowing the system to find relevant content even when exact keywords don't match.

## Maximal Marginal Relevance (MMR)

- Ensures search results aren't repetitive by balancing relevance with diversity.
- If you have 10 very similar documents that match a query, MMR will pick the best one and then look for other relevant but different perspectives, rather than showing you the same information 10 times.

## Contextual Compression Components:

### Cohere Re-ranking

- A specialized model that takes the initial search results and re-orders them by actually reading and understanding both the query and the content.
- Similar to having a human assistant who reads through search results and puts the most relevant ones at the top, but automated.

### Top-N Filtering

- Takes the re-ranked results and keeps only the most relevant ones (typically the top 5).
- This is like having an executive summary instead of a full report, ensuring the AI only works with the most important information.

### Citation Preservation

- Keeps track of where each piece of information came from in your documents, including file names and page numbers.
- Works like a reference manager in a Word document, automatically maintaining source information and allowing for proper attribution in responses.
