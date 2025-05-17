# app/services/tools/retrieve_docs.py

from typing import Dict, List, Any, Type
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
    name: str = "retrieve_docs"
    description: str = "Searches the knowledge base for relevant information based on a query"
    args_schema: Type[BaseModel] = RetrieveDocsInput
    
    # Declare vector_store as a Pydantic field
    vector_store: VectorStore = Field(..., description="The vector store instance")
    
    # Allow arbitrary types for the vector_store field
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store: VectorStore, **kwargs):
        # Pass vector_store as a keyword argument to the parent __init__
        super().__init__(vector_store=vector_store, **kwargs)
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Run the tool to retrieve documents."""
        docs = self.vector_store.search(query)
        return {"docs": docs, "query": query}