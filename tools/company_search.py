import os
from typing import List, Optional
from pydantic import PrivateAttr
from langchain.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from core import tracer

class CompanyVectorStore:

    def __init__(
        self,
        pdf_folder: str = "./company_pdfs",
        persist_directory: str = "./chroma_db",
        collection_name: str = "company_financials",
    ):
        self.pdf_folder = pdf_folder
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = OpenAIEmbeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.documents = self._load_pdfs()
        self.retriever = self._build_retriever()

    def _load_or_create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            print("[LOG] Loading existing Chroma DB...")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        else:
            print("[LOG] Chroma DB not found. Loading PDFs and creating DB...")
            documents = self._load_pdfs()
            chunks = self._split_documents(documents)
            vectorstore = Chroma.from_documents(
                chunks,
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )
            vectorstore.persist()
            print("[LOG] Chroma DB created and persisted.")
            return vectorstore
    
    def _load_pdfs(self):
        documents = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_folder, filename))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["page"] = doc.metadata.get("page", "Unknown")
                documents.extend(docs)
        print(f"[LOG] Loaded {len(documents)} documents from PDFs.")
        return documents

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[LOG] Split into {len(chunks)} chunks.")
        return chunks

    # V1 - Basic similarity search
    def similarity_search(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)
    
    # V2 - Ensemble / hybrid retirver with keywords (BM25), Similariy and MMR (Maximal marginal relevance - Don’t give me 5 copies of the same answer.)
    # V2 - Search results are ranked using Cohere ranking based on how well they answer the actual query.
    def _build_retriever(self):
        print("[LOG] Building hybrid retriever with BM25, MMR, and embeddings...")
        retriever_bm25 = BM25Retriever.from_documents(self.documents, search_kwargs={"k": 20})
        retriever_vanilla = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        retriever_mmr = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
            weights=[0.4, 0.4, 0.2],
        )

        compressor = CohereRerank(top_n=5, model="rerank-english-v3.0")

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )
        return compression_retriever

    @tracer.tool(name="company_search")
    def retrieve(self, query: str):
        return self.retriever.invoke(query)
    
class CompanySearchTool(BaseTool):
    name: str = "company_search"
    description: str = (
        "Search within Rakuten's internal financial documents to answer specific questions "
        "about Rakuten's financials, such as revenue, profits, and reports from past years."
    )

    _vectorstore: CompanyVectorStore = PrivateAttr()

    def __init__(self, company_vectorstore: CompanyVectorStore, **kwargs):
        super().__init__(vectorstore=company_vectorstore, **kwargs)
        self._vectorstore = company_vectorstore

    def _run(self, query: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        try:
            print(f"[LOG] CompanySearchTool called with query: {query}")
            results = self._vectorstore.retrieve(query)

            print(f"\n\n****[LOG] CompanySearchTool results: {results}\n\n")
            
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []