# tests/conftest.py

import pytest
from fastapi.testclient import TestClient
import json
import os
from pathlib import Path

from app.main import app
from app.utils.vector_store import VectorStore
from app.services.tools.retrieve_docs import RetrieveDocsTool
from app.services.tools.manage_tasks import ManageTasksTool
from app.services.self_grading import SelfGradingService
from app.services.self_reflection import SelfReflectionService

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def test_knowledge_base():
    """Create a temporary knowledge base for testing."""
    # Create test directory if it doesn't exist
    test_dir = Path("./tests/data")
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a simple test knowledge base
    test_kb = [
        {
            "content": "AI is a technology that enables computers to perform tasks that typically require human intelligence.",
            "source": "Test Source 1",
            "metadata": {"category": "overview"}
        },
        {
            "content": "Machine Learning is a subset of AI focused on enabling systems to learn from data.",
            "source": "Test Source 2",
            "metadata": {"category": "overview"}
        }
    ]
    
    # Write to file
    kb_path = test_dir / "test_knowledge_base.json"
    with open(kb_path, "w") as f:
        json.dump(test_kb, f)
    
    yield str(kb_path)
    
    # Cleanup
    if os.path.exists(kb_path):
        os.remove(kb_path)

@pytest.fixture
def vector_store(test_knowledge_base):
    """Create a vector store with the test knowledge base."""
    return VectorStore(test_knowledge_base)

@pytest.fixture
def retrieve_docs_tool(vector_store):
    """Create a retrieve_docs tool with the test vector store."""
    return RetrieveDocsTool(vector_store)

@pytest.fixture
def manage_tasks_tool():
    """Create a manage_tasks tool for testing."""
    return ManageTasksTool()

@pytest.fixture
def self_grading_service():
    """Create a self-grading service for testing."""
    return SelfGradingService()

@pytest.fixture
def self_reflection_service():
    """Create a self-reflection service for testing."""
    return SelfReflectionService()