# Maven Drive Copilot - Take Home Assessment

Welcome! This is a **2-3 hour coding assessment** where you'll build the backend for a Google Drive semantic search and chat system.

Please create a Fork of this repository to continue building!

## 🎯 What You'll Build

A system that:
1. **Ingests** Google Drive files into a vector database
2. **Searches** documents using semantic search
3. **Chats** with an LLM using retrieved context (RAG)

## ✅ What's Already Done

We've built everything except the core logic:

- ✅ **Basic Frontend Setup** - React + TypeScript 
- ✅ **Basic Backend Structure** - FastAPI with all routes
- ✅ **Google OAuth Integration** - Full authentication flow
- ✅ **Type Definitions** - All Pydantic models
- ✅ **API Routes** - All endpoints wired up
- ✅ **Drive Service** - Google Drive API integration

## ⚠️ What YOU Need to Implement

You need to implement **3 core functions**:

### 1. Ingestion Service (60-90 minutes)
**File:** `backend/src/services/ingestion_service.py`

**What it should do:**
- Fetch files from Google Drive
- Extract text from different file types
- Chunk content intelligently
- Generate embeddings
- Store in vector database with metadata

**Key considerations:**
- Start with Google Docs only (easiest)
- Use simple fixed-size chunking (500-1000 chars)
- Store metadata: file_id, file_name, path, chunk_index

### 2. Search Tool (45-60 minutes)
**File:** `backend/src/tools/search_tool.py`

**Function:** `search_documents()`

**What it should do:**
- Generate embedding for search query
- Query vector database for similar chunks
- Apply folder/file filters if provided
- Format results with snippets and highlights
- Return top N results with relevance scores

**Key considerations:**
- Return 5-10 most relevant results
- Calculate relevance scores (0-1 range)
- Include file metadata in results

### 3. Chat Service (30-45 minutes)
**File:** `backend/src/services/chat_service.py`

**What it should do:**
- Use `search_documents()` to find relevant context
- Build prompt with context and conversation history (optional)
- Call LLM API to generate response
- Extract and return source citations

**Key considerations:**
- Focus on ingestion and search first

## 🚀 Quick Start

### 1. Clone and Setup (5 minutes)

```bash
# Clone the repo
git clone <repo-url>
cd MavenOA

# Run the dev script (installs everything and starts services)
./dev.sh
```

The script will:
- Install Python and Node dependencies
- Prompt you to configure `.env`
- Start both backend and frontend
- Show live logs

### 2. Configure Google OAuth (10 minutes)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **Google Drive API**
4. Create OAuth 2.0 credentials:
   - Type: Web application
   - Authorized redirect URI: `http://localhost:8000/api/auth/callback`
5. Copy Client ID and Secret to `backend/.env`:

```env
GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
GOOGLE_REDIRECT_URI=http://localhost:8000/api/auth/callback
```

6. Configure OAuth Consent Screen:
   - User Type: **External**
   - Add your email as a test user
   - Add scope: `../auth/drive.readonly`

### 3. Choose Your Tech Stack (5 minutes)

Pick one from each category and add to `.env`:

**Vector Database (choose one):**
```bash
# ChromaDB (easiest - runs locally)
pip install chromadb

# OR Pinecone (cloud-hosted)
pip install pinecone-client
# Add to .env: PINECONE_API_KEY=xxx

# OR Qdrant (self-hosted or cloud)
pip install qdrant-client
```

**Embedding Model (choose one):**
```bash
# OpenAI (best quality)
pip install openai
# Add to .env: OPENAI_API_KEY=xxx

# OR Sentence Transformers (free, local)
pip install sentence-transformers

# OR Cohere
pip install cohere
# Add to .env: COHERE_API_KEY=xxx
```

**LLM for Chat (choose one):**
```bash
# OpenAI GPT-4/3.5
# Uses OPENAI_API_KEY from above

# OR Anthropic Claude
pip install anthropic
# Add to .env: ANTHROPIC_API_KEY=xxx

# OR Cohere
# Uses COHERE_API_KEY from above
```

### 4. Start Implementing (2-3 hours)

The implementation files:

```
backend/src/
├── services/
│   ├── ingestion_service.py      # ⚠️ YOU IMPLEMENT THIS
│   └── chat_service.py           # ⚠️ YOU IMPLEMENT THIS
└── tools/
    └── search_tool.py            # ⚠️ YOU IMPLEMENT THIS
```

Each file has:
- ✅ Function signatures
- ✅ Type hints
- ✅ Detailed TODO comments
- ✅ Helper function stubs


## 🧪 Testing Your Implementation

### 1. Create Test Files in Google Drive

Create a few test documents:
- `Project Plan.docx` - "We are building a Drive copilot for Maven"
- `Meeting Notes.docx` - "Discussed the take-home assessment timeline"
- `Budget.docx` - "Engineering budget is $100k"

### 2. Test Ingestion

1. Open http://localhost:3000
2. Click "Connect Google Drive"
3. Authorize the app
4. Click "Start Ingestion"
5. Watch the progress bar
6. Check logs to verify chunks are being stored

### 3. Test Search

1. In the chat interface, type: "What are we building?"
2. Verify relevant results appear
3. Check that source attribution is correct
4. Try filtering by folder/file

### 4. Test Chat (if implemented)

1. Ask: "What's the engineering budget?"
2. Verify it finds the Budget document
3. Check that sources are cited
4. Try follow-up questions

## 📊 Evaluation Criteria

We'll evaluate based on:

### 1. Functionality (40%)
- ✅ Does ingestion work end-to-end?
- ✅ Does search return relevant results?
- ✅ Are sources properly attributed?
- ✅ Does filtering work (folder/file)?

### 2. Code Quality (30%)
- ✅ Clean, readable code
- ✅ Proper error handling
- ✅ Good function/variable names
- ✅ Appropriate abstractions

### 3. Retrieval Quality (20%)
- ✅ Relevance of search results
- ✅ Quality of snippets/highlights
- ✅ Handling of edge cases

### 4. Execution (10%)
- ✅ Time management
- ✅ What you accomplished in 2-3 hours
- ✅ Clear commit messages


## ⏱️ Time Management Suggestions

**Total: 3 hours**

### Setup (30 minutes)
- ✅ Clone repo and read docs (10 min)
- ✅ Configure Google OAuth (10 min)
- ✅ Choose and install vector DB/embeddings (10 min)

### Implementation (90-120 minutes)
- ✅ Ingestion Service (60-90 min)
  - File fetching and text extraction (20 min)
  - Chunking logic (15 min)
  - Embedding generation (15 min)
  - Vector DB storage (20 min)
  - Testing (10 min)

- ✅ Search Tool (45-60 min)
  - Query embedding (10 min)
  - Vector search (15 min)
  - Result formatting (15 min)
  - Testing (15 min)

- ✅ Chat Service (30-45 min)
  - Prompt building (15 min)
  - LLM integration (15 min)
  - Testing (10 min)

### Testing & Polish (15-30 minutes)
- ✅ End-to-end testing (15 min)
- ✅ Bug fixes and polish (15 min)

## 🚨 Common Pitfalls

1. **Over-engineering chunking** - Start simple! Fixed-size chunks work fine
2. **Forgetting metadata** - Store file_id, file_name, path with each chunk
3. **Not testing early** - Test ingestion with 1 file before processing all
4. **Ignoring error handling** - At least catch and log exceptions
5. **Spending too long on chat** - Focus on ingestion and search first

## 📚 Helpful Resources

### Vector Databases
- [ChromaDB Docs](https://docs.trychroma.com/) - Easiest to set up
- [Pinecone Docs](https://docs.pinecone.io/)
- [Qdrant Docs](https://qdrant.tech/documentation/)

### Embeddings
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)

### LLMs
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude](https://docs.anthropic.com/)

### Google Drive API
- [Drive API Reference](https://developers.google.com/drive/api/v3/reference)

## 📦 Submission

When you're done:

1. **Commit your code** with clear commit messages
2. **Push to GitHub**
3. **Email us** at yuv2bindal@gmail.com and sultan.shayaan@gmail.com with:
   - Link to your repository
   - Hours spent
   - Brief description of your approach (2-3 sentences)
   - Tech stack choices (vector DB, embeddings, LLM)
   - Any challenges or trade-offs
   - What you'd improve with more time

## ❓ FAQ

**Q: What if I can't finish in 2-3 hours?**
A: Submit what you have! We value quality over completeness. Show us your best work.

**Q: Can I use AI coding assistants (GitHub Copilot, Claude, etc.)?**
A: Yes! But make sure you understand the code. We'll discuss your implementation.

**Q: Which vector DB should I use?**
A: ChromaDB is easiest for local development. Pinecone is great for cloud. Or surprise us with an even better DB.

**Q: Do I need to implement chat with streaming?**
A: It's optional. Focus on ingestion and search first. Chat with streaming is a bonus. Otherwise, just a simple chat is alright!

**Q: Can I modify the frontend or services?**
A: You can, but focus on implementing the tools. The architecture is already set up.

**Q: What about authentication persistence?**
A: Current implementation keeps credentials in memory. That's fine for this assessment.

**Q: How do I handle large files?**
A: Start simple - just handle Google Docs. If time permits, add PDF support.

**Q: Should I write tests?**
A: Manual testing is fine. Automated tests are a bonus if you have time.

## 🎯 Success Looks Like

At minimum (for a passing submission):
- ✅ Ingestion works for Google Docs
- ✅ Search returns relevant results
- ✅ Sources are properly attributed
- ✅ Code is clean and readable

Bonus points for:
- ⭐ Handling multiple file types
- ⭐ Smart chunking strategies
- ⭐ Chat implementation
- ⭐ Great result snippets/highlights
- ⭐ Robust error handling

## 🆘 Need Help?

If you run into issues:

1. **Check the logs** - `logs/backend.log` and `logs/frontend.log`
2. **Read the TODOs** - Each file has detailed implementation guidance
3. **Check the examples** - Code comments show example structures
4. **Email us** - We're happy to clarify requirements or help with setup

---

**Good luck!** We're excited to see what you build. Take your time, write clean code, and have fun! 🚀
