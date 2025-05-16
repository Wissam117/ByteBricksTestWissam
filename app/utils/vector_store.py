# app/utils/vector_store.py

import json
import os
from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, knowledge_base_path: str, embeddings=None):
        self.knowledge_base_path = knowledge_base_path
        
        # Use provided embeddings or default to OpenAI if available, otherwise use free alternative
        if embeddings:
            self.embeddings = embeddings
        elif os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings()
        else:
            # Use a free alternative like sentence-transformers
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Load documents from knowledge base and create vector store."""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.knowledge_base_path}")
        
        with open(self.knowledge_base_path, 'r') as f:
            knowledge_base = json.load(f)
        
        documents = []
        for doc in knowledge_base:
            content = doc.get("content", "")
            source = doc.get("source", "")
            metadata = doc.get("metadata", {})
            metadata["source"] = source
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="ai_concierge_docs"
        )
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents given a query."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "metadata": doc.metadata
            }
            for doc in docs
        ]