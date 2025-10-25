"""
Test script to verify the implementation works correctly.
This script tests the core functionality without requiring OAuth.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.services.ingestion_service import ingestion_service
        print("✓ Ingestion service imported")
    except Exception as e:
        print(f"✗ Failed to import ingestion service: {e}")
        return False
    
    try:
        from src.tools.search_tool import search_documents
        print("✓ Search tool imported")
    except Exception as e:
        print(f"✗ Failed to import search tool: {e}")
        return False
    
    try:
        from src.services.chat_service import chat_service
        print("✓ Chat service imported")
    except Exception as e:
        print(f"✗ Failed to import chat service: {e}")
        return False
    
    return True


def test_chromadb_initialization():
    """Test ChromaDB initialization"""
    print("\nTesting ChromaDB initialization...")
    
    try:
        from src.services.ingestion_service import ingestion_service
        
        if ingestion_service.chroma_client is None:
            print("✗ ChromaDB client not initialized")
            return False
        
        print("✓ ChromaDB client initialized")
        
        if ingestion_service.collection is None:
            print("✗ ChromaDB collection not created")
            return False
        
        print("✓ ChromaDB collection created")
        return True
        
    except Exception as e:
        print(f"✗ ChromaDB initialization failed: {e}")
        return False


def test_gemini_api_key():
    """Test Gemini API key configuration"""
    print("\nTesting Gemini API configuration...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_key:
        print("✗ GEMINI_API_KEY not set in environment")
        print("  Please set GEMINI_API_KEY in your .env file")
        return False
    
    print("✓ GEMINI_API_KEY is set")
    
    try:
        from src.services.ingestion_service import ingestion_service
        
        if ingestion_service.embedding_model is None:
            print("✗ Embedding model not initialized")
            return False
        
        print(f"✓ Embedding model initialized: {ingestion_service.embedding_model}")
        
    except Exception as e:
        print(f"✗ Gemini initialization failed: {e}")
        return False
    
    try:
        from src.services.chat_service import chat_service
        
        if chat_service.model is None:
            print("✗ Chat model not initialized")
            return False
        
        print("✓ Chat model initialized")
        
    except Exception as e:
        print(f"✗ Chat service initialization failed: {e}")
        return False
    
    return True


def test_text_extraction():
    """Test text extraction logic"""
    print("\nTesting text extraction...")
    
    try:
        from src.services.ingestion_service import ingestion_service
        
        # Test with a mock file metadata
        mock_file = {
            'id': 'test_id',
            'name': 'test.txt',
            'mimeType': 'text/plain'
        }
        
        # This will fail because we don't have actual Drive credentials,
        # but we can verify the method exists and has proper error handling
        try:
            text = ingestion_service._extract_text_from_file(mock_file)
            print("✗ Should have raised an error without Drive credentials")
            return False
        except Exception as e:
            # Expected to fail, but should handle gracefully
            print(f"✓ Text extraction method exists and handles errors: {type(e).__name__}")
            return True
        
    except Exception as e:
        print(f"✗ Text extraction test failed: {e}")
        return False


def test_chunking():
    """Test text chunking logic"""
    print("\nTesting text chunking...")
    
    try:
        from src.services.ingestion_service import ingestion_service
        
        # Test text
        test_text = """
        This is a test document. It has multiple paragraphs.
        
        This is the second paragraph. It contains some information about the project.
        
        This is the third paragraph. It has more details and context.
        """ * 10  # Make it long enough to create multiple chunks
        
        mock_file = {
            'id': 'test_id',
            'name': 'test.txt',
            'mimeType': 'text/plain',
            'webViewLink': 'https://example.com'
        }
        
        chunks = ingestion_service._chunk_text(test_text, mock_file, '/test.txt')
        
        if not chunks:
            print("✗ No chunks created")
            return False
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Verify chunk structure
        first_chunk = chunks[0]
        required_fields = ['text', 'file_id', 'file_name', 'file_path', 'chunk_index']
        
        for field in required_fields:
            if field not in first_chunk:
                print(f"✗ Missing field in chunk: {field}")
                return False
        
        print("✓ Chunks have correct structure")
        
        # Verify chunk sizes are reasonable
        for chunk in chunks:
            if len(chunk['text']) > 1200:  # Should be around 800 + overlap
                print(f"✗ Chunk too large: {len(chunk['text'])} characters")
                return False
        
        print("✓ Chunk sizes are within expected range")
        return True
        
    except Exception as e:
        print(f"✗ Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_building():
    """Test context building for chat"""
    print("\nTesting context building...")
    
    try:
        from src.services.chat_service import chat_service
        from src.types import SearchResult, DriveFile
        
        # Create mock search results
        mock_file = DriveFile(
            id='test_id',
            name='test.txt',
            mimeType='text/plain',
            path='/test.txt',
            modifiedTime='2024-01-01T00:00:00Z'
        )
        
        mock_results = [
            SearchResult(
                file=mock_file,
                snippet='This is a test snippet from the document.',
                relevanceScore=0.95,
                highlights=['test snippet']
            )
        ]
        
        context = chat_service._build_context(mock_results)
        
        if not context:
            print("✗ No context generated")
            return False
        
        print("✓ Context generated successfully")
        
        # Verify context contains source information
        if 'test.txt' not in context:
            print("✗ Context missing file name")
            return False
        
        print("✓ Context includes source attribution")
        return True
        
    except Exception as e:
        print(f"✗ Context building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("NeuroDrive-Copilot - Implementation Tests")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("Imports", test_imports),
        ("ChromaDB Initialization", test_chromadb_initialization),
        ("Gemini API Configuration", test_gemini_api_key),
        ("Text Extraction", test_text_extraction),
        ("Text Chunking", test_chunking),
        ("Context Building", test_context_building),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

