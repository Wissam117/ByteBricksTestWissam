# app/services/memory.py

from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class MemoryService:
    def __init__(self):
        self.sessions: Dict[str, List[Any]] = {}
    
    def get_messages(self, session_id: str) -> List[Any]:
        """Get all messages for a specific session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the session history.
        
        Args:
            session_id: Unique identifier for the conversation session
            role: "human", "ai", or "system"
            content: The message content
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        if role == "human":
            message = HumanMessage(content=content)
        elif role == "ai":
            message = AIMessage(content=content)
        elif role == "system":
            message = SystemMessage(content=content)
        else:
            raise ValueError(f"Invalid role: {role}")
        
        self.sessions[session_id].append(message)
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a specific session."""
        self.sessions[session_id] = []