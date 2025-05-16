# app/services/tools/retrieve_docs.py

from typing import Dict, List, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.utils.vector_store import VectorStore
from app.core.config import settings

class RetrieveDocsInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant documents")

class RetrieveDocsOutput(BaseModel):
    docs: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    query: str = Field(..., description="The original search query")

class RetrieveDocsTool(BaseTool):
    name = "retrieve_docs"
    description = "Searches the knowledge base for relevant information based on a query"
    args_schema = RetrieveDocsInput
    
    def __init__(self, vector_store: VectorStore):
        super().__init__()
        self.vector_store = vector_store
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Run the tool to retrieve documents."""
        docs = self.vector_store.search(query)
        return {"docs": docs, "query": query}