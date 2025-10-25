# RAG Configuration Summary

## Updated Configuration (Latest)

### 1. **Chat Model**
- **Model**: `gpt-4o` (Latest GPT-4 Omni model)
  - Note: GPT-5 is not yet released, so using the most advanced GPT-4 variant
- **Temperature**: 0.7
- **Max Tokens**: 4096
- **Location**: `backend/src/services/chat_service.py`

### 2. **Chunking Strategy**
- **Method**: Recursive Character Text Splitter
- **Chunk Size**: 2000 characters
- **Overlap**: 800 characters
- **Separator Hierarchy** (in order of preference):
  1. `\n\n` - Paragraphs (highest priority)
  2. `\n` - Lines
  3. `. ` - Sentences (period)
  4. `! ` - Sentences (exclamation)
  5. `? ` - Sentences (question)
  6. `; ` - Clauses
  7. `, ` - Phrases
  8. ` ` - Words
  9. `""` - Characters (fallback)
- **Location**: `backend/src/services/ingestion_service.py`

**Why Recursive Splitter?**
- Better preserves semantic boundaries
- Tries to split at natural breakpoints (paragraphs → sentences → words)
- Falls back gracefully if no good split point exists
- More context-aware than fixed-size splitting

### 3. **Vector Database**
- **Database**: ChromaDB (local persistent)
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: Cosine Similarity
- **Collection**: `drive_documents`
- **Configuration**:
  ```python
  metadata={
      "description": "Google Drive documents with embeddings",
      "hnsw:space": "cosine"  # Cosine similarity
  }
  ```
- **Location**: `backend/src/services/ingestion_service.py`

**Why HNSW + Cosine?**
- **HNSW**: Fast approximate nearest neighbor search, scales well
- **Cosine**: Better for text embeddings (measures angle, not magnitude)
- **vs L2**: Cosine is more robust to document length variations

### 4. **Embedding Model**
- **Model**: OpenAI `text-embedding-3-large`
- **Dimensions**: 3072
- **Batch Size**: 100 documents per batch
- **Location**: `backend/src/services/ingestion_service.py`

### 5. **Retrieval Configuration**
- **Top-K Results**: 5 (for chat context)
- **Similarity Metric**: Cosine similarity
- **Score Calculation**: `relevance_score = 1 - cosine_distance`
  - Range: 0.0 to 1.0 (higher is better)
  - 1.0 = identical vectors
  - 0.0 = orthogonal vectors
- **Location**: `backend/src/tools/search_tool.py`

### 6. **Supported File Types**
- Google Docs → `text/plain` export
- Google Sheets → `text/csv` export
- Google Slides → `text/plain` export
- PDF files → PyPDF2 extraction
- Plain text files → Direct download

## Key Improvements Over Previous Configuration

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Chat Model | `gpt-4o-mini` | `gpt-4o` | Better reasoning, more accurate responses |
| Chunking | Fixed 800 chars | Recursive 2000 chars | Better semantic preservation |
| Overlap | 200 chars | 800 chars | More context between chunks |
| Index | Flat (default) | HNSW | Faster search at scale |
| Distance | L2 | Cosine | Better for text similarity |
| Score Range | Unbounded | 0.0-1.0 | Normalized, easier to interpret |

## Performance Considerations

### Chunking Trade-offs
- **Larger chunks (2000)**: More context per chunk, fewer total chunks
- **Higher overlap (800)**: Better continuity, but more storage
- **Recursive splitting**: Slightly slower than fixed-size, but much better quality

### HNSW Parameters (ChromaDB defaults)
- `M`: 16 (connections per node)
- `ef_construction`: 100 (construction-time accuracy)
- `ef_search`: 10 (search-time accuracy)
- Can be tuned via ChromaDB settings if needed

### Cosine vs L2
- **Cosine**: Normalized by vector length, better for varying document lengths
- **L2**: Sensitive to magnitude, can favor shorter documents
- For text embeddings, cosine is generally preferred

## Environment Variables

Required:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

Optional:
```env
CHROMA_PERSIST_DIRECTORY=./chroma_db  # Default location
ANONYMIZED_TELEMETRY=false  # Disable ChromaDB telemetry
```

## Migration Notes

If you have existing data in ChromaDB with L2 distance:

1. **Option 1**: Delete and re-ingest
   ```bash
   rm -rf ./chroma_db
   # Then run ingestion again
   ```

2. **Option 2**: Create new collection
   - Change collection name in code (e.g., `drive_documents_v2`)
   - Re-ingest data
   - Update search tool to use new collection

## Testing the Configuration

1. **Start backend**:
   ```bash
   cd backend
   source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test ingestion**:
   - Connect Google Drive via frontend
   - Start ingestion
   - Check logs for "ChromaDB collection initialized with HNSW index and cosine similarity"

3. **Test search**:
   - Query documents
   - Check relevance scores (should be 0.0-1.0 range)
   - Verify chunks are ~2000 chars

4. **Test chat**:
   - Ask questions
   - Verify GPT-4o responses (should be higher quality than mini)
   - Check source citations

## Further Optimizations (Optional)

### If you need better performance:
1. **Reduce chunk size**: 1000-1500 chars for faster embedding
2. **Reduce overlap**: 400 chars to save storage
3. **Use smaller embedding model**: `text-embedding-3-small` (cheaper, faster)
4. **Tune HNSW**: Increase `ef_search` for better accuracy

### If you need better quality:
1. **Increase top-K**: Retrieve 10-15 chunks instead of 5
2. **Add reranking**: Use cross-encoder to rerank results
3. **Hybrid search**: Combine semantic + keyword search
4. **Query expansion**: Rephrase user query multiple ways

## References

- [ChromaDB HNSW Configuration](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Recursive Character Text Splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)
- [Cosine vs L2 Distance](https://en.wikipedia.org/wiki/Cosine_similarity)

