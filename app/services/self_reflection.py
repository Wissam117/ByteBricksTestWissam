# app/services/self_reflection.py

from typing import Dict, Optional

class SelfReflectionService:
    def __init__(self):
        self.decay_factor = 0.5
        self.sessions: Dict[str, float] = {}
    
    def get_session_score(self, session_id: str) -> float:
        """Get the current feedback score for a session."""
        return self.sessions.get(session_id, 0.0)
    
    def apply_feedback(self, session_id: str, feedback_type: str) -> float:
        """
        Apply feedback to adjust the session score.
        
        Args:
            session_id: Unique identifier for the conversation session
            feedback_type: "good_answer" or "bad_answer"
            
        Returns:
            Updated cumulative score
        """
        current_score = self.get_session_score(session_id)
        
        if feedback_type == "good_answer":
            current_score += 1
        elif feedback_type == "bad_answer":
            current_score -= 1
        
        self.sessions[session_id] = current_score
        return current_score
    
    def apply_decay(self, session_id: str) -> float:
        """
        Apply decay factor to the session score after each turn.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            Updated score after decay
        """
        current_score = self.get_session_score(session_id)
        decayed_score = current_score * self.decay_factor
        self.sessions[session_id] = decayed_score
        return decayed_score
    
    def get_modified_prompt(self, session_id: str) -> Optional[str]:
        """
        Generate a modified prompt based on the cumulative score.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            Modified prompt instruction or None if no modification needed
        """
        current_score = self.get_session_score(session_id)
        
        if current_score < 0:
            return "Be more concise and cite sources explicitly."
        elif current_score > 0:
            return "Maintain current style, user is satisfied."
        
        return None