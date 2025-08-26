"""
RAG (Retrieval-Augmented Generation) system for AI Interview Coach
Manages vector database, embeddings, and document retrieval for interview questions and contexts.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import faiss
import numpy as np
import pickle
import yaml
from sentence_transformers import SentenceTransformer
import httpx

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import chroma_config, embedding_config

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Manages retrieval of relevant interview content using vector similarity search.
    Supports multiple data sources: questions, rubrics, concepts, and examples.
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 docs_dir: str = "docs",
                 embedding_model: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Initialize RAG retriever with vector database and embedding model.
        Uses configuration from environment variables by default.
        
        Args:
            data_dir: Directory for ChromaDB storage (uses config if None)
            docs_dir: Directory containing source documents
            embedding_model: SentenceTransformers model name (uses config if None)
            collection_name: ChromaDB collection name (uses config if None)
        """
        self.data_dir = Path(data_dir or chroma_config.db_path)
        self.docs_dir = Path(docs_dir)
        self.embedding_model_name = embedding_model or embedding_config.model_name
        self.collection_name = collection_name or chroma_config.collection_name
        self.device = embedding_config.device
        
        # Initialize components
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize_database()
        self._initialize_embeddings()
    
    def _initialize_database(self):
        """Initialize FAISS index and metadata storage."""
        try:
            # Create data directory if it doesn't exist
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize FAISS index - will be created when first documents are added
            self.index = None
            self.documents = []  # Store document content
            self.metadata = []   # Store document metadata
            self.ids = []        # Store document IDs
            
            # Paths for saving/loading index and metadata
            self.index_path = self.data_dir / f"{self.collection_name}.index"
            self.metadata_path = self.data_dir / f"{self.collection_name}.pkl"
            
            # Try to load existing index
            self._load_index()
            
            logger.info(f"Initialized FAISS with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize Ollama client for embeddings."""
        try:
            # Use Ollama for embeddings
            self.ollama_base_url = "http://localhost:11434"
            self.embedding_model_name = "nomic-embed-text"
            logger.info(f"Using Ollama embeddings with model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                    self.ids = data['ids']
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                logger.info("No existing index found, will create new one")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}, starting fresh")
            self.index = None
            self.documents = []
            self.metadata = []
            self.ids = []

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_path))
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'metadata': self.metadata,
                        'ids': self.ids
                    }, f)
                
                logger.info("Saved index to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using Ollama."""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model_name,
                    "prompt": text
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.warning(f"Embedding request failed with status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def load_documents(self, force_reload: bool = False):
        """
        Load and index all documents from the docs directory.
        
        Args:
            force_reload: If True, clear existing data and reload everything
        """
        if force_reload:
            logger.info("Force reload requested - clearing existing data")
            self.index = None
            self.documents = []
            self.metadata = []
            self.ids = []
        
        # Check if already has documents and not force reloading
        if len(self.documents) > 0 and not force_reload:
            logger.info(f"Index already contains {len(self.documents)} documents")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        # Load interview questions
        questions_dir = self.docs_dir / "interview_questions"
        if questions_dir.exists():
            docs = self._load_interview_questions(questions_dir)
            documents.extend([doc['content'] for doc in docs])
            metadatas.extend([doc['metadata'] for doc in docs])
            ids.extend([doc['id'] for doc in docs])
        
        # Load job rubrics
        rubrics_dir = self.docs_dir / "job_rubrics"
        if rubrics_dir.exists():
            docs = self._load_rubrics(rubrics_dir)
            documents.extend([doc['content'] for doc in docs])
            metadatas.extend([doc['metadata'] for doc in docs])
            ids.extend([doc['id'] for doc in docs])
        
        # Load concept explanations
        concepts_dir = self.docs_dir / "concepts"
        if concepts_dir.exists():
            docs = self._load_concepts(concepts_dir)
            documents.extend([doc['content'] for doc in docs])
            metadatas.extend([doc['metadata'] for doc in docs])
            ids.extend([doc['id'] for doc in docs])
        
        if documents:
            logger.info(f"Processing {len(documents)} documents...")
            
            # Get embeddings for all documents
            embeddings = []
            for i, doc in enumerate(documents):
                logger.info(f"Getting embedding for document {i+1}/{len(documents)}")
                embedding = self._get_embedding(doc)
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Failed to get embedding for document {ids[i]}, skipping")
                    continue
            
            if embeddings:
                # Convert to numpy array
                embeddings_np = np.array(embeddings, dtype=np.float32)
                
                # Create FAISS index if it doesn't exist
                if self.index is None:
                    dimension = embeddings_np.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    logger.info(f"Created new FAISS index with dimension {dimension}")
                
                # Add embeddings to index
                self.index.add(embeddings_np)
                
                # Store documents and metadata
                self.documents.extend(documents)
                self.metadata.extend(metadatas)
                self.ids.extend(ids)
                
                # Save to disk
                self._save_index()
                
                logger.info(f"Successfully indexed {len(embeddings)} documents")
            else:
                logger.error("No embeddings generated, cannot create index")
        else:
            logger.warning("No documents found to index")
    
    def _load_interview_questions(self, questions_dir: Path) -> List[Dict]:
        """Load interview questions from markdown/yaml files."""
        documents = []
        
        for file_path in questions_dir.glob("**/*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Extract metadata from filename and content
                filename_parts = file_path.stem.split('_')
                
                # Extract role (everything except the last part which should be level)
                if len(filename_parts) >= 2:
                    # Check if last part is a level
                    potential_level = filename_parts[-1].lower()
                    if potential_level in ["junior", "mid", "senior", "principal"]:
                        role = "_".join(filename_parts[:-1])
                        level = potential_level
                    else:
                        role = "_".join(filename_parts)
                        level = "mid"  # Default level
                else:
                    role = filename_parts[0] if filename_parts else "general"
                    level = "mid"  # Default level
                
                # Parse individual questions from the markdown content
                parsed_questions = self._parse_questions_from_markdown(content)
                
                for i, question_data in enumerate(parsed_questions):
                    documents.append({
                        'id': f"question_{file_path.stem}_{i}",
                        'content': question_data['content'],
                        'metadata': {
                            'type': 'interview_question',
                            'role': role,
                            'level': level,
                            'source_file': str(file_path),
                            'difficulty': question_data.get('difficulty', 'medium'),
                            'topics': question_data.get('topics', []),
                            'question_number': i + 1
                        }
                    })
                
            except Exception as e:
                logger.error(f"Error loading question file {file_path}: {e}")
        
        return documents
    
    def _parse_questions_from_markdown(self, content: str) -> List[Dict]:
        """Parse individual questions from markdown content."""
        questions = []
        
        import re
        
        # Try multiple question patterns to handle different formats
        patterns = [
            r'### Question:(.*?)(?=### Question:|$)',  # New format: ### Question:
            r'## Question \d+:(.*?)(?=## Question \d+:|$)',  # Old format: ## Question 1:
        ]
        
        question_matches = []
        for pattern in patterns:
            question_matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if question_matches:
                break
        
        if not question_matches:
            # Fallback: if no structured questions found, try to extract by ## headers
            sections = re.split(r'\n## ', content)
            for i, section in enumerate(sections[1:], 1):  # Skip first section (title)
                if section.strip():
                    # Extract the question text (first paragraph after header)
                    lines = section.strip().split('\n')
                    title = lines[0] if lines else f"Question {i}"
                    
                    # Find the actual question (first paragraph)
                    question_text = ""
                    for line in lines[1:]:
                        line = line.strip()
                        if line and not line.startswith('**') and not line.startswith('Topic'):
                            question_text = line
                            break
                    
                    if question_text:
                        questions.append({
                            'content': f"{title}\n{question_text}",
                            'difficulty': 'medium',
                            'topics': []
                        })
        else:
            # Parse structured questions
            for i, question_match in enumerate(question_matches, 1):
                question_content = question_match.strip()
                
                # Extract difficulty and topics from metadata
                difficulty = 'medium'  # default
                topics = []
                topic_name = ""
                question_type = ""
                
                # Parse Topic (single topic)
                if '**Topic:**' in question_content:
                    topic_match = re.search(r'\*\*Topic\*\*:\s*([^\n]+)', question_content)
                    if topic_match:
                        topic_name = topic_match.group(1).strip()
                        topics = [topic_name]
                
                # Parse Difficulty
                if '**Difficulty:**' in question_content:
                    difficulty_match = re.search(r'\*\*Difficulty\*\*:\s*(\w+)', question_content)
                    if difficulty_match:
                        difficulty = difficulty_match.group(1).lower()
                
                # Parse Type
                if '**Type:**' in question_content:
                    type_match = re.search(r'\*\*Type\*\*:\s*([^\n]+)', question_content)
                    if type_match:
                        question_type = type_match.group(1).strip()
                
                # Fallback: Parse Topics (multiple topics)
                if not topics and '**Topics:**' in question_content:
                    topics_match = re.search(r'\*\*Topics\*\*:\s*([^\n]+)', question_content)
                    if topics_match:
                        topics = [t.strip() for t in topics_match.group(1).split(',')]
                
                # Extract the main question text (first few lines before **Expected Answer** or section headers)
                question_lines = question_content.split('\n')
                main_question = []
                for line in question_lines:
                    line = line.strip()
                    # Stop at expected answer, topics, or section headers
                    if (line.startswith('**Expected Answer**') or 
                        line.startswith('**Topics**') or 
                        line.startswith('##')):
                        break
                    if line and not line.startswith('**'):
                        main_question.append(line)
                
                if main_question:
                    questions.append({
                        'content': '\n'.join(main_question),
                        'difficulty': difficulty,
                        'topics': topics
                    })
        
        return questions
    
    def _load_rubrics(self, rubrics_dir: Path) -> List[Dict]:
        """Load job rubrics and evaluation criteria."""
        documents = []
        
        for file_path in rubrics_dir.glob("**/*.yaml"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    rubric_data = yaml.safe_load(f)
                
                # Convert rubric to searchable text
                content = self._rubric_to_text(rubric_data)
                
                documents.append({
                    'id': f"rubric_{file_path.stem}_{len(documents)}",
                    'content': content,
                    'metadata': {
                        'type': 'job_rubric',
                        'role': rubric_data.get('role', 'general'),
                        'level': rubric_data.get('level', 'all'),
                        'source_file': str(file_path)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error loading rubric file {file_path}: {e}")
        
        return documents
    
    def _load_concepts(self, concepts_dir: Path) -> List[Dict]:
        """Load technical concept explanations."""
        documents = []
        
        for file_path in concepts_dir.glob("**/*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Extract topic from filename
                topic = file_path.stem.replace('_', ' ').replace('-', ' ')
                
                documents.append({
                    'id': f"concept_{file_path.stem}_{len(documents)}",
                    'content': content,
                    'metadata': {
                        'type': 'concept',
                        'topic': topic,
                        'source_file': str(file_path)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error loading concept file {file_path}: {e}")
        
        return documents
    
    def _rubric_to_text(self, rubric_data: Dict) -> str:
        """Convert rubric YAML data to searchable text."""
        text_parts = []
        
        if 'title' in rubric_data:
            text_parts.append(f"Title: {rubric_data['title']}")
        
        if 'description' in rubric_data:
            text_parts.append(f"Description: {rubric_data['description']}")
        
        if 'criteria' in rubric_data:
            text_parts.append("Evaluation Criteria:")
            for criterion in rubric_data['criteria']:
                if isinstance(criterion, dict):
                    for key, value in criterion.items():
                        text_parts.append(f"- {key}: {value}")
                else:
                    text_parts.append(f"- {criterion}")
        
        return '\n'.join(text_parts)
    
    async def search_questions(self, 
                             role: str, 
                             level: str = "mid", 
                             topic: Optional[str] = None,
                             difficulty: Optional[str] = None,
                             limit: int = 5) -> List[Dict]:
        """
        Search for relevant interview questions based on criteria using FAISS.
        
        Args:
            role: Target role (e.g., "software_engineer", "genai_engineer")
            level: Experience level (junior, mid, senior, principal)
            topic: Specific topic area (optional)
            difficulty: Question difficulty (optional)
            limit: Maximum number of results
            
        Returns:
            List of relevant questions with metadata
        """
        try:
            # Check if index exists and has data
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents indexed for search")
                return []
            
            # Build search query
            query_parts = [f"interview questions for {role}", f"{level} level"]
            if topic:
                query_parts.append(topic)
            if difficulty:
                query_parts.append(f"{difficulty} difficulty")
            
            query = " ".join(query_parts)
            
            # Get embedding for query
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            # Search FAISS index
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_vector, min(limit * 2, len(self.documents)))
            
            # Filter results based on metadata criteria with fallback levels
            questions = []
            fallback_questions = []
            
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                doc_metadata = self.metadata[idx]
                
                if doc_metadata.get('type') == 'interview_question' and doc_metadata.get('role') == role:
                    # Calculate relevance score (convert L2 distance to similarity)
                    distance = distances[0][i]
                    relevance_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    question_data = {
                        'content': self.documents[idx],
                        'metadata': doc_metadata,
                        'relevance_score': relevance_score
                    }
                    
                    # Exact level match
                    if doc_metadata.get('level') == level:
                        # Additional filter for difficulty if specified
                        if not difficulty or doc_metadata.get('difficulty') == difficulty:
                            questions.append(question_data)
                    else:
                        # Store as fallback if level doesn't match
                        if not difficulty or doc_metadata.get('difficulty') == difficulty:
                            fallback_questions.append(question_data)
                
                # Stop when we have enough exact matches
                if len(questions) >= limit:
                    break
            
            # If we don't have enough exact level matches, use fallbacks
            if len(questions) < limit:
                needed = limit - len(questions)
                questions.extend(fallback_questions[:needed])
                if len(fallback_questions) > 0:
                    logger.info(f"Using {len(fallback_questions[:needed])} fallback questions from different levels")
            
            logger.info(f"Found {len(questions)} questions for {role} {level}")
            return questions
            
        except Exception as e:
            logger.error(f"Error searching questions: {e}")
            return []
    
    async def search_concepts(self, 
                            query: str, 
                            limit: int = 3) -> List[Dict]:
        """
        Search for relevant concept explanations using FAISS.
        
        Args:
            query: Search query for concepts
            limit: Maximum number of results
            
        Returns:
            List of relevant concepts with metadata
        """
        try:
            # Check if index exists and has data
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents indexed for concept search")
                return []
            
            # Get embedding for query
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            # Search FAISS index
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_vector, min(limit * 2, len(self.documents)))
            
            # Filter results for concepts
            concepts = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                doc_metadata = self.metadata[idx]
                
                # Apply metadata filter for concepts
                if doc_metadata.get('type') == 'concept':
                    # Calculate relevance score (convert L2 distance to similarity)
                    distance = distances[0][i]
                    relevance_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    concepts.append({
                        'content': self.documents[idx],
                        'metadata': doc_metadata,
                        'relevance_score': relevance_score
                    })
                
                # Stop when we have enough concepts
                if len(concepts) >= limit:
                    break
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    async def get_rubric(self, role: str, level: str = "all") -> Optional[Dict]:
        """
        Get evaluation rubric for a specific role and level using FAISS.
        
        Args:
            role: Target role
            level: Experience level
            
        Returns:
            Rubric data if found
        """
        try:
            # Check if index exists and has data
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents indexed for rubric search")
                return None
            
            # Get embedding for query
            query = f"evaluation rubric for {role} {level}"
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return None
            
            # Search FAISS index for best match
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_vector, len(self.documents))
            
            # Find the best matching rubric
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                doc_metadata = self.metadata[idx]
                
                # Check if this is a rubric for the specified role
                if (doc_metadata.get('type') == 'job_rubric' and 
                    doc_metadata.get('role') == role):
                    
                    return {
                        'content': self.documents[idx],
                        'metadata': doc_metadata
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting rubric: {e}")
            return None
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the current FAISS collection."""
        try:
            count = len(self.documents)
            
            types = {}
            roles = {}
            levels = {}
            
            # Analyze metadata
            for metadata in self.metadata:
                doc_type = metadata.get('type', 'unknown')
                types[doc_type] = types.get(doc_type, 0) + 1
                
                if 'role' in metadata:
                    role = metadata['role']
                    roles[role] = roles.get(role, 0) + 1
                
                if 'level' in metadata:
                    level = metadata['level']
                    levels[level] = levels.get(level, 0) + 1
            
            return {
                'total_documents': count,
                'document_types': types,
                'roles': roles,
                'levels': levels,
                'index_dimensions': self.index.d if self.index else 0,
                'index_total': self.index.ntotal if self.index else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'error': str(e)}

# Utility functions for CLI usage
def initialize_rag(data_dir: str = "data/vector_db", 
                   docs_dir: str = "docs") -> RAGRetriever:
    """Initialize RAG system and load documents."""
    rag = RAGRetriever(data_dir=data_dir, docs_dir=docs_dir)
    rag.load_documents()
    return rag

async def main():
    """CLI interface for testing RAG functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Retriever for AI Interview Coach")
    parser.add_argument("--action", choices=["load", "search", "stats"], 
                       default="stats", help="Action to perform")
    parser.add_argument("--role", default="software_engineer", 
                       help="Role for question search")
    parser.add_argument("--level", default="mid", 
                       help="Level for question search")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--force-reload", action="store_true", 
                       help="Force reload all documents")
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG
    rag = RAGRetriever()
    
    if args.action == "load":
        rag.load_documents(force_reload=args.force_reload)
        print("Documents loaded successfully")
    
    elif args.action == "search":
        if args.query:
            concepts = await rag.search_concepts(args.query)
            print(f"Found {len(concepts)} relevant concepts:")
            for concept in concepts:
                print(f"- {concept['metadata'].get('topic', 'Unknown')} "
                      f"(score: {concept['relevance_score']:.3f})")
        
        questions = await rag.search_questions(args.role, args.level)
        print(f"Found {len(questions)} questions for {args.role} {args.level}:")
        for i, question in enumerate(questions[:3], 1):
            print(f"{i}. {question['content'][:100]}...")
    
    elif args.action == "stats":
        stats = rag.get_collection_stats()
        print("Collection Statistics:")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Document types: {stats.get('document_types', {})}")
        print(f"Roles: {stats.get('roles', {})}")
        print(f"Levels: {stats.get('levels', {})}")

if __name__ == "__main__":
    asyncio.run(main())