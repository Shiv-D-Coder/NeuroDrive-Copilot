"""
Search Tool - Document search implementation using ChromaDB and OpenAI

This tool handles the core search functionality:
- Query embedding generation using OpenAI text-embedding-3-large
- Vector database search using ChromaDB
- Metadata filtering
- Result formatting
"""

from typing import List, Optional
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from ..types import SearchResult, DriveFile
import logging

logger = logging.getLogger(__name__)

# Initialize ChromaDB client (shared with ingestion service)
persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
chroma_client = chromadb.Client(Settings(
    persist_directory=persist_directory,
    anonymized_telemetry=False
))

# Get collection (with HNSW index and cosine similarity)
try:
    collection = chroma_client.get_collection(name="drive_documents")
    logger.info("Connected to ChromaDB collection with HNSW index and cosine similarity")
except Exception as e:
    logger.warning(f"Collection not found: {e}")
    collection = None

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        embedding_model = 'text-embedding-3-large'
        logger.info("OpenAI client initialized for search")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client for search: {e}")
        openai_client = None
        embedding_model = None
else:
    logger.warning("OPENAI_API_KEY not set. Search will not work.")
    openai_client = None
    embedding_model = None


async def search_documents(
    query: str,
    folder_id: Optional[str] = None,
    file_id: Optional[str] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search for documents using semantic search.

    Args:
        query: Search query text
        folder_id: Optional folder to restrict search
        file_id: Optional specific file to search within
        limit: Maximum number of results to return

    Returns:
        List of SearchResult with relevance scores and snippets
    """
    if not collection:
        logger.error("ChromaDB collection not initialized")
        return []
    
    if not openai_client or not embedding_model:
        logger.error("OpenAI client not initialized")
        return []
    
    try:
        # Step 1: Generate embedding for the query
        logger.info(f"Searching for: {query}")
        
        response = openai_client.embeddings.create(
            model=embedding_model,
            input=[query],
            encoding_format="float"
        )
        
        query_embedding = response.data[0].embedding
        
        # Step 2: Build filter for ChromaDB
        where_filter = None
        if file_id:
            where_filter = {"file_id": file_id}
        # Note: folder filtering would require folder_id to be stored in metadata
        
        # Step 3: Search in ChromaDB
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )
        
        # Step 4: Format results
        formatted_results = []
        
        if not search_results['ids'] or not search_results['ids'][0]:
            logger.info("No results found")
            return []
        
        # Extract results
        ids = search_results['ids'][0]
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        
        for i in range(len(ids)):
            metadata = metadatas[i]
            document = documents[i]
            distance = distances[i]
            
            # Convert distance to relevance score (0-1, higher is better)
            # With cosine similarity, ChromaDB returns cosine distance (1 - cosine_similarity)
            # So distance ranges from 0 (identical) to 2 (opposite)
            # Convert to similarity score: 1 - distance gives us cosine similarity
            relevance_score = max(0.0, min(1.0, 1.0 - distance))
            
            # Create snippet with highlighting
            snippet = _create_snippet(document, query)
            
            # Create DriveFile object
            drive_file = DriveFile(
                id=metadata['file_id'],
                name=metadata['file_name'],
                mimeType=metadata['mime_type'],
                path=metadata['file_path'],
                modifiedTime="",  # Not stored in chunks
                size=None,
                webViewLink=metadata.get('web_view_link', '')
            )
            
            # Create SearchResult
            search_result = SearchResult(
                file=drive_file,
                snippet=snippet,
                relevanceScore=relevance_score,
                highlights=_extract_highlights(document, query)
            )
            
            formatted_results.append(search_result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


def _create_snippet(text: str, query: str, max_length: int = 200) -> str:
    """
    Create a snippet from text, trying to include query terms.
    
    Args:
        text: Full text
        query: Search query
        max_length: Maximum snippet length
    
    Returns:
        Snippet string
    """
    if len(text) <= max_length:
        return text
    
    # Try to find query terms in text
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Find first occurrence of any query word
    query_words = query_lower.split()
    best_pos = 0
    
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = pos
            break
    
    # Create snippet around the found position
    start = max(0, best_pos - max_length // 2)
    end = min(len(text), start + max_length)
    
    snippet = text[start:end]
    
    # Add ellipsis if truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet.strip()


def _extract_highlights(text: str, query: str, max_highlights: int = 3) -> List[str]:
    """
    Extract highlighted phrases from text that match query terms.
    
    Args:
        text: Full text
        query: Search query
        max_highlights: Maximum number of highlights
    
    Returns:
        List of highlight strings
    """
    highlights = []
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Split query into words
    query_words = query_lower.split()
    
    # Find sentences containing query words
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()
        
        # Check if sentence contains any query word
        for word in query_words:
            if word in sentence_lower and sentence not in highlights:
                highlights.append(sentence)
                break
        
        if len(highlights) >= max_highlights:
            break
    
    return highlights
