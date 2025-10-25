"""
Chat Routes - Candidates implement the service logic
These routes are complete, but call ChatService methods that need implementation
"""

from fastapi import APIRouter, HTTPException
from ..services.chat_service import chat_service
from ..types import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat interactions with Drive documents.

    This endpoint is COMPLETE. Candidates implement the
    chat_service.chat() method.

    The service should:
    - Use retrieval service to find relevant documents
    - Build context from search results
    - Generate response using LLM
    - Return response with source citations
    """
    try:
        response = await chat_service.chat(request)
        return response
    except NotImplementedError as e:
        raise HTTPException(
            status_code=501,
            detail="Chat functionality not yet implemented. Please implement chat_service.chat()"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
