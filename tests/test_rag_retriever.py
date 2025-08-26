"""
Tests for the RAG retriever system
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock chromadb and sentence_transformers before importing
sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.config'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from scripts.rag_retriever import RAGRetriever

class TestRAGRetriever:
    """Test RAG retriever functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "vector_db"
        self.docs_dir = Path(self.temp_dir) / "docs"
        
        # Create test directories
        self.data_dir.mkdir(parents=True)
        self.docs_dir.mkdir(parents=True)
        
        # Create a test document
        test_doc = self.docs_dir / "test_questions.md"
        test_doc.write_text("""
# Test Interview Questions

## Software Engineer Questions

### Question 1
What is the time complexity of binary search?

**Answer:** O(log n)

### Question 2
Explain the difference between a stack and a queue.

**Answer:** Stack is LIFO, queue is FIFO.
        """)

    def test_initialization_with_defaults(self):
        """Test RAGRetriever initialization with default values."""
        with patch.dict('os.environ', {
            'CHROMA_DB_PATH': str(self.data_dir),
            'CHROMA_COLLECTION_NAME': 'test_collection',
            'EMBEDDING_MODEL': 'test-model',
            'EMBEDDING_DEVICE': 'cpu'
        }):
            retriever = RAGRetriever(docs_dir=str(self.docs_dir))
            
            assert retriever.data_dir == self.data_dir
            assert retriever.docs_dir == self.docs_dir
            assert retriever.collection_name == 'test_collection'
            assert retriever.embedding_model_name == 'test-model'
            assert retriever.device == 'cpu'

    def test_initialization_with_custom_values(self):
        """Test RAGRetriever initialization with custom values."""
        custom_data_dir = str(self.data_dir / "custom")
        custom_collection = "custom_collection"
        custom_model = "custom-model"
        
        retriever = RAGRetriever(
            data_dir=custom_data_dir,
            docs_dir=str(self.docs_dir),
            embedding_model=custom_model,
            collection_name=custom_collection
        )
        
        assert retriever.data_dir == Path(custom_data_dir)
        assert retriever.collection_name == custom_collection
        assert retriever.embedding_model_name == custom_model

    @patch('scripts.rag_retriever.chromadb')
    @patch('scripts.rag_retriever.SentenceTransformer')
    def test_database_initialization(self, mock_sentence_transformer, mock_chromadb):
        """Test database initialization."""
        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        retriever = RAGRetriever(
            data_dir=str(self.data_dir),
            docs_dir=str(self.docs_dir)
        )
        
        # Verify ChromaDB was initialized
        mock_chromadb.PersistentClient.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
        
        # Verify sentence transformer was initialized
        mock_sentence_transformer.assert_called_once()
        
        # Verify components are set
        assert retriever.client == mock_client
        assert retriever.collection == mock_collection
        assert retriever.embedding_model == mock_model

    @patch('scripts.rag_retriever.chromadb')
    @patch('scripts.rag_retriever.SentenceTransformer')
    def test_get_collection_stats(self, mock_sentence_transformer, mock_chromadb):
        """Test getting collection statistics."""
        # Mock the collection with count method
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        retriever = RAGRetriever(
            data_dir=str(self.data_dir),
            docs_dir=str(self.docs_dir)
        )
        
        stats = retriever.get_collection_stats()
        
        assert "document_count" in stats
        assert "collection_name" in stats
        assert stats["document_count"] == 42

    @patch('scripts.rag_retriever.chromadb')
    @patch('scripts.rag_retriever.SentenceTransformer')
    def test_search_questions(self, mock_sentence_transformer, mock_chromadb):
        """Test searching for questions."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]  # Mock embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Mock the collection query
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['What is binary search?', 'Explain recursion.']],
            'metadatas': [[{'role': 'software_engineer', 'topic': 'algorithms'}, 
                          {'role': 'software_engineer', 'topic': 'algorithms'}]],
            'distances': [[0.1, 0.3]]
        }
        
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        retriever = RAGRetriever(
            data_dir=str(self.data_dir),
            docs_dir=str(self.docs_dir)
        )
        
        # Test search
        results = retriever.search_questions(
            role="software_engineer",
            topic="algorithms",
            query="binary search",
            num_results=2
        )
        
        # Verify search was performed
        mock_model.encode.assert_called_once_with("binary search")
        mock_collection.query.assert_called_once()
        
        # Verify results
        assert len(results) == 2
        assert results[0]['content'] == 'What is binary search?'
        assert results[0]['metadata']['role'] == 'software_engineer'

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class TestRAGRetrieverIntegration:
    """Integration tests for RAG retriever (requires actual dependencies)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_loading_integration(self):
        """Test actual document loading (requires real dependencies)."""
        # This test should only run if the actual dependencies are available
        try:
            import chromadb
            import sentence_transformers
        except ImportError:
            pytest.skip("Integration test requires chromadb and sentence_transformers")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "vector_db"
            docs_dir = Path(temp_dir) / "docs"
            
            # Create test directories and documents
            docs_dir.mkdir(parents=True)
            test_doc = docs_dir / "test.md"
            test_doc.write_text("# Test Document\nThis is a test.")
            
            # Initialize retriever
            retriever = RAGRetriever(
                data_dir=str(data_dir),
                docs_dir=str(docs_dir),
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Load documents
            await retriever.load_documents()
            
            # Verify collection has documents
            stats = retriever.get_collection_stats()
            assert stats["document_count"] > 0

if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])