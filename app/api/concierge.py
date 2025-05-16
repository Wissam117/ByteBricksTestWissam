# app/api/concierge.py

from typing import Any, Dict
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

from app.api.auth import get_current_user
from app.models.user import User
from app.services.concierge import ConciergeService

router = APIRouter()

# Initialize concierge service
concierge_service = ConciergeService()

# Rate limiting utility
rate_limits = {}

def check_rate_limit(user_id: str, limit: int = 30) -> bool:
    """
    Check if the user has exceeded their rate limit.
    
    Args:
        user_id: User identifier
        limit: Maximum requests per minute
        
    Returns:
        True if rate limit is not exceeded, False otherwise
    """
    current_time = time.time()
    minute_window = int(current_time / 60)
    
    # Initialize or reset rate limit window
    if user_id not in rate_limits or rate_limits[user_id]["window"] != minute_window:
        rate_limits[user_id] = {"window": minute_window, "count": 0}
    
    # Check if limit is exceeded
    if rate_limits[user_id]["count"] >= limit:
        return False
    
    # Increment request count
    rate_limits[user_id]["count"] += 1
    return True

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Chat with the AI concierge.
    """
    # Check rate limit
    if not check_rate_limit(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 30 requests per minute.",
        )
    
    # Process chat message
    result = await concierge_service.chat(current_user.id, request.message)
    
    return {"response": result["response"]}

async def stream_response(user_id: str, message: str):
    """Generate a streaming response for chat."""
    # This is a simplified implementation since actual streaming
    # would require modifications to the concierge service
    result = await concierge_service.chat(user_id, message)
    response = result["response"]
    
    # Simulate streaming by yielding chunks
    for i in range(0, len(response), 10):
        chunk = response[i:i+10]
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.05)
    
    yield "data: [DONE]\n\n"

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Chat with the AI concierge with streaming response.
    """
    # Check rate limit
    if not check_rate_limit(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 30 requests per minute.",
        )
    
    return StreamingResponse(
        stream_response(current_user.id, request.message),
        media_type="text/event-stream"
    )

# Optional: Voice interface endpoint
@router.post("/voice")
async def voice_interface(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Process voice input and return voice response.
    This is a placeholder for the bonus voice interface.
    """
    # This would require integrating with Vosk/Whisper for STT
    # and pyttsx3/edge-tts for TTS
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Voice interface is not implemented yet.",
    )