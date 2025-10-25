"""
Ingestion Service - Implementation with ChromaDB and Gemini

This service:
1. Connects to Google Drive using the DriveService
2. Fetches all files and their content
3. Processes and chunks the content appropriately
4. Generates embeddings using Gemini API
5. Stores chunks and embeddings in ChromaDB
"""

from typing import List, Dict, Optional
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from .drive_service import drive_service
from ..types import IngestionStatus
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service responsible for ingesting Google Drive files into ChromaDB.
    Uses OpenAI text-embedding-3-large for embeddings.
    """

    def __init__(self):
        # Initialize ChromaDB client
        persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection with HNSW index and cosine similarity
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="drive_documents",
                metadata={
                    "description": "Google Drive documents with embeddings",
                    "hnsw:space": "cosine"  # Use cosine similarity instead of L2
                }
            )
            logger.info("ChromaDB collection initialized with HNSW index and cosine similarity")
        except Exception as e:
            logger.error(f"Error creating ChromaDB collection: {e}")
            self.collection = None
        
        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.embedding_model = 'text-embedding-3-large'
                logger.info("OpenAI client initialized with text-embedding-3-large")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
                self.embedding_model = None
        else:
            logger.warning("OPENAI_API_KEY not set. Embeddings will not work.")
            self.openai_client = None
            self.embedding_model = None

        # Status tracking
        self.is_ingesting = False
        self.total_files = 0
        self.processed_files = 0
        self.current_file = None
        self.error = None

    async def start_ingestion(self) -> Dict[str, str]:
        """
        Start the ingestion process.
        
        Steps:
        1. Set ingestion status flags
        2. Fetch all files from Google Drive
        3. For each file, extract text, chunk, embed, and store
        4. Handle errors and update status
        """
        self.is_ingesting = True
        self.processed_files = 0
        self.error = None
        
        try:
            # Fetch all files from Google Drive
            logger.info("Fetching files from Google Drive...")
            files = drive_service.list_files()
            
            # Filter only supported file types (exclude folders)
            supported_files = [
                f for f in files 
                if f.get('mimeType') != 'application/vnd.google-apps.folder'
            ]
            
            self.total_files = len(supported_files)
            logger.info(f"Found {self.total_files} files to process")
            
            # Process each file
            for file_metadata in supported_files:
                try:
                    self.current_file = file_metadata['name']
                    logger.info(f"Processing: {self.current_file}")
                    
                    # Extract text from file
                    text = self._extract_text_from_file(file_metadata)
                    
                    if not text or len(text.strip()) < 10:
                        logger.warning(f"Skipping {self.current_file} - no text content")
                        self.processed_files += 1
                        continue
                    
                    # Build file path
                    file_path = drive_service.build_file_path(
                        file_metadata['id'],
                        file_metadata['name'],
                        file_metadata.get('parents')
                    )
                    
                    # Chunk the text
                    chunks = self._chunk_text(text, file_metadata, file_path)
                    
                    if not chunks:
                        logger.warning(f"No chunks created for {self.current_file}")
                        self.processed_files += 1
                        continue
                    
                    # Generate embeddings
                    embeddings = self._generate_embeddings(chunks)
                    
                    # Store in vector database
                    self._store_in_vector_db(chunks, embeddings)
                    
                    logger.info(f"Successfully processed {self.current_file} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_metadata['name']}: {str(e)}")
                    # Continue with next file
                
                self.processed_files += 1
            
            logger.info(f"Ingestion completed: {self.processed_files}/{self.total_files} files")
            return {"message": f"Ingestion completed successfully. Processed {self.processed_files} files."}
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Ingestion failed: {str(e)}")
            raise
        finally:
            self.is_ingesting = False
            self.current_file = None

    def get_status(self) -> IngestionStatus:
        """
        Get the current ingestion status.

        This method is COMPLETE - no implementation needed.
        """
        return IngestionStatus(
            isIngesting=self.is_ingesting,
            totalFiles=self.total_files,
            processedFiles=self.processed_files,
            currentFile=self.current_file,
            error=self.error
        )

    def _extract_text_from_file(self, file_metadata: dict) -> str:
        """
        Extract text content from a file based on its MIME type.

        Supports:
        - Google Docs: Export as plain text
        - Google Sheets: Export as CSV
        - PDFs: Extract text using PyPDF2
        - Plain text: Direct download

        Args:
            file_metadata: File metadata from Google Drive

        Returns:
            Extracted text content
        """
        mime_type = file_metadata.get('mimeType', '')
        file_id = file_metadata['id']
        
        try:
            # Google Docs
            if mime_type == 'application/vnd.google-apps.document':
                content = drive_service.export_google_doc(file_id, 'text/plain')
                return content.decode('utf-8')
            
            # Google Sheets
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                content = drive_service.export_google_doc(file_id, 'text/csv')
                return content.decode('utf-8')
            
            # Google Slides
            elif mime_type == 'application/vnd.google-apps.presentation':
                content = drive_service.export_google_doc(file_id, 'text/plain')
                return content.decode('utf-8')
            
            # PDF files
            elif mime_type == 'application/pdf':
                try:
                    from PyPDF2 import PdfReader
                    import io
                    
                    content = drive_service.download_file(file_id)
                    pdf_file = io.BytesIO(content)
                    reader = PdfReader(pdf_file)
                    
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    return text
                except Exception as e:
                    logger.error(f"Error extracting PDF text: {e}")
                    return ""
            
            # Plain text files
            elif mime_type.startswith('text/'):
                content = drive_service.download_file(file_id)
                return content.decode('utf-8', errors='ignore')
            
            else:
                logger.warning(f"Unsupported MIME type: {mime_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_metadata['name']}: {e}")
            return ""

    def _chunk_text(self, text: str, file_metadata: dict, file_path: str) -> List[Dict[str, any]]:
        """
        Chunk text using recursive character text splitter.

        Strategy:
        - Recursive splitting with multiple separators (paragraphs, sentences, words)
        - 2000 character chunks with 800 character overlap
        - Preserves semantic boundaries better than fixed-size chunking
        - Include metadata for each chunk

        Args:
            text: Full text content
            file_metadata: File metadata for context
            file_path: Full path of the file

        Returns:
            List of chunks with metadata
        """
        # Chunk parameters
        chunk_size = 2000  # characters per chunk
        overlap = 800  # overlap between chunks for context
        
        chunks = []
        
        # Clean the text
        text = text.strip()
        
        if len(text) == 0:
            return chunks
        
        # Recursive character text splitter - try separators in order of preference
        separators = [
            "\n\n",  # Paragraphs (highest priority)
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Sentences
            "? ",    # Sentences
            "; ",    # Clauses
            ", ",    # Phrases
            " ",     # Words
            ""       # Characters (fallback)
        ]
        
        text_chunks = self._recursive_split(text, separators, chunk_size, overlap)
        
        # Create chunk objects with metadata
        start_char = 0
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text.strip(),
                "file_id": file_metadata['id'],
                "file_name": file_metadata['name'],
                "file_path": file_path,
                "mime_type": file_metadata.get('mimeType', ''),
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": start_char + len(chunk_text),
                "web_view_link": file_metadata.get('webViewLink', '')
            })
            start_char += len(chunk_text) - overlap  # Account for overlap
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
        """
        Recursively split text using multiple separators.
        
        Args:
            text: Text to split
            separators: List of separators to try in order
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        # Try each separator
        for i, separator in enumerate(separators):
            if separator == "":
                # Fallback: split by character
                return self._split_with_overlap(text, chunk_size, overlap)
            
            if separator in text:
                # Split by this separator
                splits = text.split(separator)
                
                # Reconstruct chunks respecting chunk_size
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Add separator back (except for empty string)
                    split_with_sep = split + separator if separator else split
                    
                    if len(current_chunk) + len(split_with_sep) <= chunk_size:
                        current_chunk += split_with_sep
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If single split is too large, recursively split it
                        if len(split_with_sep) > chunk_size:
                            sub_chunks = self._recursive_split(
                                split_with_sep, 
                                separators[i+1:], 
                                chunk_size, 
                                overlap
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split_with_sep
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Add overlap between chunks
                return self._add_overlap(chunks, overlap)
        
        return [text]
    
    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        
        return chunks
    
    def _add_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of chunks
            overlap: Number of characters to overlap
            
        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1 or overlap == 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # Get overlap from previous chunk
            prev_chunk = overlapped_chunks[-1]
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            
            # Prepend overlap to current chunk
            overlapped_chunks.append(overlap_text + chunks[i])
        
        return overlapped_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using OpenAI text-embedding-3-large.

        Args:
            chunks: List of text chunks

        Returns:
            List of embedding vectors
        """
        if not self.openai_client or not self.embedding_model:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        
        embeddings = []
        batch_texts = [chunk['text'] for chunk in chunks]
        
        # Process in batches to avoid rate limits
        batch_size = 100
        
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i + batch_size]
            
            try:
                # Generate embeddings using OpenAI
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                    encoding_format="float"
                )
                
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(batch_texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Add zero vectors as fallback (3072 dimensions for text-embedding-3-large)
                for _ in batch:
                    embeddings.append([0.0] * 3072)
        
        logger.info(f"Generated {len(embeddings)} embeddings using OpenAI {self.embedding_model}")
        return embeddings

    def _store_in_vector_db(self, chunks: List[Dict[str, any]], embeddings: List[List[float]]) -> None:
        """
        Store chunks and embeddings in ChromaDB.

        Args:
            chunks: List of chunks with metadata
            embeddings: Corresponding embedding vectors
        """
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID for each chunk
            chunk_id = f"{chunk['file_id']}_chunk_{chunk['chunk_index']}"
            ids.append(chunk_id)
            
            # Document text
            documents.append(chunk['text'])
            
            # Metadata (ChromaDB requires all values to be strings, ints, floats, or bools)
            metadatas.append({
                'file_id': chunk['file_id'],
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'mime_type': chunk['mime_type'],
                'chunk_index': chunk['chunk_index'],
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char'],
                'web_view_link': chunk.get('web_view_link', '')
            })
            
            embeddings_list.append(embedding)
        
        try:
            # Upsert to ChromaDB (will update if ID exists, insert if new)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
            
        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")
            raise


# Global instance
ingestion_service = IngestionService()
