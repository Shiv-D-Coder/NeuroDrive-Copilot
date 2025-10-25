"""
Chat Service - Implementation with OpenAI API

This service:
1. Uses search tool to find relevant chunks
2. Builds context from search results
3. Generates intelligent responses using OpenAI GPT
4. Cites sources clearly in responses
"""

from typing import List
import os
from openai import OpenAI
from ..types import ChatRequest, ChatResponse, SearchResult
from ..tools.search_tool import search_documents
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service responsible for handling chat interactions with context from Google Drive.
    Uses OpenAI GPT for chat responses.
    """

    def __init__(self):
        # Defer OpenAI client initialization to runtime to avoid startup failures
        self.openai_client = None
        self.model = 'gpt-4o'  # Latest GPT-4 model (GPT-5 not yet released)

    def _ensure_openai(self) -> None:
        if self.openai_client is not None:
            return
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not set. Chat will not work.")
            return
        try:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized for chat")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client for chat: {e}")
            self.openai_client = None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat message and return a response with sources.

        Steps:
        1. Use search tool to find relevant chunks
        2. Build context from retrieved chunks
        3. Construct prompt with context and conversation history
        4. Generate response using Gemini
        5. Return response with source citations

        Args:
            request: ChatRequest with message and optional context

        Returns:
            ChatResponse with generated message and source citations
        """
        # Ensure OpenAI is ready
        self._ensure_openai()
        if not self.openai_client or not self.model:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing chat message: {request.message}")
            sources = await search_documents(
                query=request.message,
                folder_id=request.folder_id,
                file_id=request.file_id,
                limit=5
            )
            
            # Step 2: Build context from sources
            context = self._build_context(sources)
            
            # Step 3: Generate response
            response_text = await self._generate_response(
                message=request.message,
                context=context,
                history=request.conversation_history
            )
            
            # Step 4: Return with sources
            return ChatResponse(
                message=response_text,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            # Return error response
            return ChatResponse(
                message=f"I encountered an error while processing your request: {str(e)}",
                sources=[]
            )

    def _build_context(self, sources: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            sources: List of SearchResult objects

        Returns:
            Formatted context string
        """
        if not sources:
            return "No relevant documents found in your Google Drive."
        
        context_parts = []
        
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}] From '{source.file.name}' ({source.file.path}):\n"
                f"{source.snippet}\n"
            )
        
        return "\n\n".join(context_parts)

    async def _generate_response(self, message: str, context: str, history: List = None) -> str:
        """
        Generate LLM response given message, context, and history using OpenAI GPT.

        Args:
            message: User's question
            context: Context from retrieved documents
            history: Optional conversation history

        Returns:
            Generated response text
        """
        # Build messages for OpenAI chat completion
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions about documents in a Google Drive. Use the provided context to answer questions accurately. Always cite your sources by mentioning the document name. If the answer isn't in the context, say so clearly. Be concise and helpful."""
            }
        ]
        
        # Add conversation history if provided
        if history:
            for msg in history[-5:]:  # Last 5 messages to avoid token limits
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current question with context
        user_message = f"Context from Google Drive:\n{context}\n\nUser Question: {message}"
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"


# Global instance
chat_service = ChatService()
