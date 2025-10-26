"""
Search Tool - Hybrid search implementation with vector + keyword search and reranking

This tool handles the core search functionality:
- Hybrid retrieval: Vector search (semantic) + BM25 (keyword)
- Cross-encoder reranking for improved relevance
- Query embedding generation using OpenAI text-embedding-3-large
- Vector database search using ChromaDB
- Metadata filtering
- Result formatting
"""

from typing import List, Optional, Dict, Tuple
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rank_bm25 import BM25Okapi
# Lazy import for CrossEncoder to avoid startup crashes if HF deps are missing
reranker = None
def _ensure_reranker():
    global reranker
    if reranker is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder reranker initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize reranker: {e}")
        reranker = None
from ..types import SearchResult, DriveFile
import logging

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (used across functions)
RRF_K = 60

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

# Note: reranker is initialized lazily via _ensure_reranker()


async def search_documents(
    query: str,
    folder_id: Optional[str] = None,
    file_id: Optional[str] = None,
    limit: int = 10,
    relevance_threshold: float = 0.3
) -> List[SearchResult]:
    """
    Hybrid search for documents using vector search + BM25 keyword search + reranking.
    
    Only returns documents that meet the relevance threshold to avoid irrelevant results.

    Process:
    1. Vector search (semantic) - retrieve top 2*limit results
    2. BM25 keyword search - retrieve top 2*limit results
    3. Merge and deduplicate results using Reciprocal Rank Fusion (RRF)
    4. Rerank merged results using cross-encoder
    5. Filter by relevance threshold
    6. Return relevant results (dynamic count based on quality)

    Args:
        query: Search query text
        folder_id: Optional folder to restrict search
        file_id: Optional specific file to search within
        limit: Maximum number of results to return
        relevance_threshold: Minimum relevance score (0-1) to include result (default: 0.3)

    Returns:
        List of SearchResult with relevance scores and snippets (only relevant docs)
    """
    if not collection:
        logger.error("ChromaDB collection not initialized")
        return []
    
    if not openai_client or not embedding_model:
        logger.error("OpenAI client not initialized")
        return []
    
    try:
        logger.info(f"Hybrid search for: {query}")
        
        # Build filter for ChromaDB
        where_filter = None
        if file_id:
            where_filter = {"file_id": file_id}
        
        # Step 1: Vector Search (Semantic)
        vector_results = await _vector_search(query, where_filter, limit * 2)
        logger.info(f"Vector search found {len(vector_results)} results")
        
        # Step 2: BM25 Keyword Search
        bm25_results = await _bm25_search(query, where_filter, limit * 2)
        logger.info(f"BM25 search found {len(bm25_results)} results")
        
        # Step 3: Merge results using Reciprocal Rank Fusion (RRF)
        merged_results = _reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K)
        logger.info(f"Merged to {len(merged_results)} unique results")
        
        # Step 4: Rerank using cross-encoder
        _ensure_reranker()
        if reranker and len(merged_results) > 0:
            final_results = _rerank_results(query, merged_results)
            logger.info(f"Reranked {len(final_results)} results")
        else:
            # Normalize RRF scores to 0-1 scale so UI shows meaningful percentages
            # Theoretical max RRF score if a doc is rank 1 in both lists: 2 / (k + 1)
            rrf_max = 2.0 / (RRF_K + 1.0)
            final_results = []
            for doc_id, document, metadata, score in merged_results:
                normalized_score = min(1.0, score / rrf_max) if rrf_max > 0 else 0.0
                final_results.append((doc_id, document, metadata, normalized_score))
        
        # Step 5: Filter by relevance threshold to exclude irrelevant docs
        relevant_results = [
            result for result in final_results 
            if result[3] >= relevance_threshold  # result[3] is the score
        ]
        
        logger.info(f"Filtered to {len(relevant_results)} results above threshold {relevance_threshold}")
        
        # Step 6: Return only relevant results (dynamic count, not hardcoded)
        # Take up to 'limit' results, but only if they're actually relevant
        formatted_results = []
        for result in relevant_results[:limit]:
            doc_id, document, metadata, score = result
            
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
                relevanceScore=score,
                highlights=_extract_highlights(document, query)
            )
            
            formatted_results.append(search_result)
        
        logger.info(f"Returning {len(formatted_results)} final results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []


async def _vector_search(
    query: str,
    where_filter: Optional[Dict] = None,
    limit: int = 20
) -> List[Tuple[str, str, Dict, float]]:
    """
    Perform vector search using ChromaDB.
    
    Returns:
        List of tuples: (doc_id, document_text, metadata, score)
    """
    try:
        # Generate embedding for the query
        response = openai_client.embeddings.create(
            model=embedding_model,
            input=[query],
            encoding_format="float"
        )
        
        query_embedding = response.data[0].embedding
        
        # Search in ChromaDB
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )
        
        if not search_results['ids'] or not search_results['ids'][0]:
            return []
        
        # Format results
        results = []
        ids = search_results['ids'][0]
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        
        for i in range(len(ids)):
            # Convert cosine distance to similarity score
            score = max(0.0, min(1.0, 1.0 - distances[i]))
            results.append((ids[i], documents[i], metadatas[i], score))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []


async def _bm25_search(
    query: str,
    where_filter: Optional[Dict] = None,
    limit: int = 20
) -> List[Tuple[str, str, Dict, float]]:
    """
    Perform BM25 keyword search on all documents in collection.
    
    Returns:
        List of tuples: (doc_id, document_text, metadata, score)
    """
    try:
        # Fetch all documents from ChromaDB (or filtered subset)
        all_docs = collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        if not all_docs['ids'] or len(all_docs['ids']) == 0:
            return []
        
        # Prepare documents for BM25
        ids = all_docs['ids']
        documents = all_docs['documents']
        metadatas = all_docs['metadatas']
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        
        # Create results with scores
        results = []
        for i, score in enumerate(bm25_scores):
            if score > 0:  # Only include documents with non-zero scores
                # Normalize BM25 score to 0-1 range relative to max for this query
                normalized_score = (score / max_bm25) if max_bm25 > 0 else 0.0
                results.append((ids[i], documents[i], metadatas[i], normalized_score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[3], reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}")
        return []


def _reciprocal_rank_fusion(
    vector_results: List[Tuple[str, str, Dict, float]],
    bm25_results: List[Tuple[str, str, Dict, float]],
    k: int = 60
) -> List[Tuple[str, str, Dict, float]]:
    """
    Merge results from vector and BM25 search using Reciprocal Rank Fusion.
    
    RRF formula: score(d) = sum(1 / (k + rank(d)))
    
    Args:
        vector_results: Results from vector search
        bm25_results: Results from BM25 search
        k: Constant for RRF (typically 60)
    
    Returns:
        Merged and sorted results
    """
    # Create a dictionary to store RRF scores
    rrf_scores: Dict[str, Tuple[str, Dict, float]] = {}
    
    # Add vector search results
    for rank, (doc_id, document, metadata, score) in enumerate(vector_results):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = (document, metadata, 0.0)
        doc, meta, current_score = rrf_scores[doc_id]
        rrf_scores[doc_id] = (doc, meta, current_score + 1.0 / (k + rank + 1))
    
    # Add BM25 search results
    for rank, (doc_id, document, metadata, score) in enumerate(bm25_results):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = (document, metadata, 0.0)
        doc, meta, current_score = rrf_scores[doc_id]
        rrf_scores[doc_id] = (doc, meta, current_score + 1.0 / (k + rank + 1))
    
    # Convert to list and sort by RRF score
    merged = [(doc_id, doc, meta, score) for doc_id, (doc, meta, score) in rrf_scores.items()]
    merged.sort(key=lambda x: x[3], reverse=True)
    
    return merged


def _rerank_results(
    query: str,
    results: List[Tuple[str, str, Dict, float]]
) -> List[Tuple[str, str, Dict, float]]:
    """
    Rerank results using cross-encoder model.
    
    Args:
        query: Search query
        results: List of search results
    
    Returns:
        Reranked results
    """
    if not reranker or len(results) == 0:
        return results
    
    try:
        # Prepare query-document pairs
        pairs = [(query, doc) for _, doc, _, _ in results]
        
        # Get cross-encoder scores
        scores = reranker.predict(pairs)
        
        # Normalize scores to 0-1 range
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = [(s - min_score) / score_range for s in scores]
        else:
            normalized_scores = [0.5] * len(scores)
        
        # Create reranked results with new scores
        reranked = []
        for i, (doc_id, doc, meta, _) in enumerate(results):
            reranked.append((doc_id, doc, meta, normalized_scores[i]))
        
        # Sort by reranked score
        reranked.sort(key=lambda x: x[3], reverse=True)
        
        return reranked
        
    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        return results


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
