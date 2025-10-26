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
            # Step 1: Retrieve relevant documents with quality filtering
            logger.info(f"Processing chat message: {request.message}")
            sources = await search_documents(
                query=request.message,
                folder_id=request.folder_id,
                file_id=request.file_id,
                limit=15,  # Retrieve more chunks for comprehensive answers
                relevance_threshold=0.30  # Slightly lower threshold to get more context
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
        Build comprehensive context string from search results.
        Groups chunks by file to provide better context.

        Args:
            sources: List of SearchResult objects

        Returns:
            Formatted context string with complete information
        """
        if not sources:
            return "No relevant documents found in your Google Drive."
        
        # Group sources by file to provide complete context
        file_contexts = {}
        for source in sources:
            file_name = source.file.name
            if file_name not in file_contexts:
                file_contexts[file_name] = {
                    'path': source.file.path,
                    'snippets': []
                }
            file_contexts[file_name]['snippets'].append(source.snippet)
        
        # Build comprehensive context
        context_parts = []
        for i, (file_name, data) in enumerate(file_contexts.items(), 1):
            # Combine all snippets from the same file
            combined_content = "\n\n".join(data['snippets'])
            context_parts.append(
                f"[Document {i}] '{file_name}' ({data['path']}):\n"
                f"{combined_content}\n"
            )
        
        return "\n\n---\n\n".join(context_parts)

    async def _generate_response(self, message: str, context: str, history: List = None) -> str:
        """
        Generate LLM response using Chain-of-Thought prompting for multi-step reasoning.

        The model uses internal CoT reasoning but only outputs the final answer to the user.

        Args:
            message: User's question
            context: Context from retrieved documents
            history: Optional conversation history

        Returns:
            Generated response text (final answer only, no reasoning steps shown)
        """
        # Build messages for OpenAI chat completion with improved CoT prompting
        messages = [
            {
                "role": "system",
                "content": """You are an expert assistant that provides comprehensive, detailed answers about documents in Google Drive.

CRITICAL INSTRUCTIONS:

1. **THOROUGH INFORMATION EXTRACTION**:
   - Read ALL provided document content carefully and completely
   - Extract EVERY relevant piece of information, not just the first thing you see
   - If multiple chunks from the same document are provided, synthesize them together
   - Look for: numbers, dates, names, amounts, percentages, timelines, lists, categories
   - Don't stop at the first answer - check if there's more information in other parts

2. **COMPREHENSIVE ANSWERS**:
   - Provide DETAILED, COMPLETE answers (aim for 3-6 paragraphs for substantial questions)
   - Include ALL relevant information from the documents, not just a summary
   - For questions about specific documents (e.g., "what's in the budget file"), provide a COMPLETE overview
   - List specific details: numbers, items, categories, dates, etc.
   - Structure longer answers with clear organization (use natural paragraphs, not bullet points unless specifically requested)

3. **SYNTHESIS ACROSS CHUNKS**:
   - If information is spread across multiple chunks from the same document, combine them
   - Present a unified, coherent answer that brings all pieces together
   - Don't treat chunks as separate - they're parts of the same document

4. **OUTPUT FORMAT**:
   - Provide ONLY your final comprehensive answer (no "Step 1", "Step 2" labels)
   - Use natural, flowing paragraphs
   - Cite sources naturally: "According to the Budget document..." or "The Budget file shows..."
   - Be thorough but readable - organize information logically

5. **DIRECT QUESTIONS ABOUT FILES**:
   - For questions like "what's in [filename]" or "tell me about [filename]":
     * Provide a COMPLETE summary of the document's contents
     * Include ALL major sections, categories, and key information
     * List specific numbers, amounts, dates, and details
     * Be comprehensive - this is the user asking for the full picture

6. **SPECIFIC INFORMATION QUERIES**:
   - For questions like "what is my budget" or "what are the costs":
     * Extract ALL relevant numbers and details
     * Provide complete breakdowns if available
     * Include totals, subtotals, categories, and line items
     * Don't just mention that information exists - provide the actual data

7. **ACCURACY & COMPLETENESS**:
   - Never make up information not in the documents
   - If you find relevant information, include ALL of it, not just part
   - Be precise with numbers, dates, and specific details
   - If information seems incomplete, say so, but provide everything that IS available

8. **RELEVANCE CHECK**:
   - ONLY use information from documents that actually relate to the question
   - If NO documents are relevant, clearly state: "I couldn't find information about [topic] in the provided documents."
   - But if documents ARE relevant, extract EVERYTHING relevant from them

REMEMBER: Users want COMPLETE, DETAILED answers. Don't be brief when there's more information available. Extract and present ALL relevant content from the documents."""
            }
        ]
        
        # Add conversation history if provided
        if history:
            for msg in history[-5:]:  # Last 5 messages to avoid token limits
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current question with improved prompt
        user_message = f"""Context from Google Drive:
{context}

User Question: {message}

Instructions: 
- Read ALL the provided content from the documents above
- Extract EVERY piece of relevant information (numbers, details, categories, etc.)
- If multiple chunks from the same document are shown, synthesize them together
- Provide a COMPREHENSIVE, DETAILED answer with all the information available
- Cite documents naturally in your response
- Be thorough - include all relevant details, not just a brief summary"""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Generate response using OpenAI with CoT prompting
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
