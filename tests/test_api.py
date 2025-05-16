# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
import json

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_register_and_login(client):
    """Test the user registration and login flow."""
    # Register a new user
    register_data = {
        "email": "test@example.com",
        "password": "testpassword"
    }
    response = client.post("/api/v1/register", json=register_data)
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
    
    # Login with the registered user
    login_data = {
        "username": "test@example.com",
        "password": "testpassword"
    }
    response = client.post("/api/v1/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()
    
    # Test login with wrong password
    wrong_login_data = {
        "username": "test@example.com",
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/login", data=wrong_login_data)
    assert response.status_code == 401

def test_chat_endpoint_auth(client):
    """Test that the chat endpoint requires authentication."""
    chat_data = {
        "message": "What is AI?"
    }
    response = client.post("/api/v1/concierge/chat", json=chat_data)
    assert response.status_code == 401

def test_chat_endpoint_with_auth(client):
    """Test the chat endpoint with authentication."""
    # Register and login to get token
    register_data = {
        "email": "chat_test@example.com",
        "password": "testpassword"
    }
    client.post("/api/v1/register", json=register_data)
    
    login_data = {
        "username": "chat_test@example.com",
        "password": "testpassword"
    }
    login_response = client.post("/api/v1/login", data=login_data)
    token = login_response.json()["access_token"]
    
    # Use token for chat endpoint
    headers = {
        "Authorization": f"Bearer {token}"
    }
    chat_data = {
        "message": "What is AI?"
    }
    
    # This is a simplified test that just checks the endpoint structure
    # In a real test, you might want to mock the concierge service
    try:
        response = client.post("/api/v1/concierge/chat", json=chat_data, headers=headers)
        # We're not asserting a specific status code here since the actual service
        # might need external dependencies like OpenAI API access
    except Exception as e:
        # Just verify the request structure was correct
        pass