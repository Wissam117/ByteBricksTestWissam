# tests/test_tools.py

import pytest
import json

def test_retrieve_docs_tool(retrieve_docs_tool):
    """Test the retrieve_docs tool."""
    # Test simple query
    result = retrieve_docs_tool._run("What is AI?")
    
    # Check structure
    assert "docs" in result
    assert "query" in result
    assert result["query"] == "What is AI?"
    
    # Check docs returned
    assert len(result["docs"]) > 0
    assert "content" in result["docs"][0]
    assert "source" in result["docs"][0]

def test_manage_tasks_tool(manage_tasks_tool):
    """Test the manage_tasks tool."""
    # Test adding a task
    task_data = {
        "title": "Test Demo",
        "when": "Tomorrow at 3PM",
        "description": "Demo of AI technology"
    }
    
    result = manage_tasks_tool._run(
        action="add", 
        task=task_data,
        user_id="test_user"
    )
    
    assert "added" in result.lower()
    assert "Test Demo" in result
    
    # Test listing tasks
    result = manage_tasks_tool._run(
        action="list",
        user_id="test_user"
    )
    
    assert isinstance(result, dict)
    assert "tasks" in result
    assert len(result["tasks"]) == 1
    assert result["tasks"][0].title == "Test Demo"
    assert result["tasks"][0].when == "Tomorrow at 3PM"
    
    # Test invalid action
    result = manage_tasks_tool._run(
        action="invalid",
        user_id="test_user"
    )
    
    assert "invalid" in result.lower()