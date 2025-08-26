"""
Question asking module for AI Interview Coach
Orchestrates the interview process using MCP context and RAG retrieval.

TUTORIAL: This script is the "Interview Conductor" - it manages the entire interview flow.
Think of it as a smart interview coordinator that:
1. Tracks interview sessions (like a session manager)
2. Selects appropriate questions using RAG
3. Adapts difficulty based on progress
4. Generates natural question presentations using AI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.rag_retriever import RAGRetriever
from src.config import ollama_config
import httpx

logger = logging.getLogger(__name__)

@dataclass
class InterviewSession:
    """
    TUTORIAL: Data class that represents an active interview session.
    
    Think of this as a "smart folder" that keeps track of everything happening
    in one interview session. It's like having a clipboard with all the 
    interview details, questions asked, and current progress.
    
    Why use @dataclass?
    - Automatically generates __init__, __repr__, __eq__ methods
    - Type hints make code self-documenting
    - Immutable by default (safer than regular classes)
    """
    session_id: str                    # Unique identifier like "interview_20250822_143052"
    role: str                          # "software_engineer", "genai_engineer", "data_scientist", etc.
    level: str                         # "junior", "mid", "senior", "principal"
    candidate_name: Optional[str] = None    # Optional candidate name
    start_time: datetime = None        # When interview started
    questions_asked: List[Dict] = None # History of all questions asked
    current_question_number: int = 0   # Which question we're on (1, 2, 3...)
    total_questions: int = 5           # How many questions total
    session_context: Dict = None       # Any additional context/notes
    
    def __post_init__(self):
        """
        TUTORIAL: This runs after the dataclass __init__ method.
        Used to set default values for mutable types (lists, dicts).
        
        Why not set defaults directly? 
        Because Python shares mutable defaults between instances!
        """
        if self.questions_asked is None:
            self.questions_asked = []
        if self.session_context is None:
            self.session_context = {}
        if self.start_time is None:
            self.start_time = datetime.now()

@dataclass
class Question:
    """
    TUTORIAL: Represents a single interview question with all its metadata.
    
    This is like a "smart question card" that knows:
    - What the question is
    - Who it's for (role/level)
    - How difficult it is
    - What concepts it tests
    - What follow-up questions to ask
    """
    question_id: str                   # Unique identifier
    content: str                       # The actual question text
    role: str                          # Target role
    level: str                         # Target level
    topic: str                         # "algorithms", "statistics", "ml_modeling", etc.
    difficulty: str                    # "easy", "medium", "hard"
    expected_concepts: List[str]       # What concepts should be mentioned
    follow_up_prompts: List[str]       # Potential follow-up questions
    context: Optional[str] = None      # Additional context/scenario

class QuestionAsker:
    """
    TUTORIAL: The main "Interview Conductor" class.
    
    This is the brain of the interview system. It:
    1. Manages multiple interview sessions simultaneously
    2. Uses RAG to find relevant questions
    3. Uses AI (Ollama) to present questions naturally
    4. Adapts difficulty based on progress
    5. Tracks session state and history
    
    Architecture Pattern: This follows the "Orchestrator" pattern - it doesn't
    do the work itself, but coordinates other components (RAG, AI, contexts).
    """
    
    def __init__(self, 
                 rag_retriever: Optional[RAGRetriever] = None,
                 contexts_dir: str = "contexts"):
        """
        TUTORIAL: Initialize the question asker with its dependencies.
        
        Dependency Injection Pattern:
        - Accept RAGRetriever as parameter (testable, flexible)
        - Load MCP contexts from YAML files
        - Set up Ollama client for AI generation
        """
        self.rag = rag_retriever or RAGRetriever()
        self.contexts_dir = Path(contexts_dir)
        self.interviewer_context = self._load_context("interviewer.context.yaml")
        self.sessions: Dict[str, InterviewSession] = {}  # Active sessions storage
        
        # Initialize Ollama client for AI text generation
        self.ollama_client = httpx.AsyncClient(
            base_url=ollama_config.base_url,
            timeout=300.0  # Increase timeout to 5 minutes for llama3.2:3b
        )
    
    def _load_context(self, context_file: str) -> Dict:
        """
        TUTORIAL: Load MCP context from YAML file.
        
        MCP (Model Context Protocol) contexts are like "personality files"
        that tell the AI how to behave. This loads the interviewer's
        personality and behavior guidelines.
        """
        try:
            context_path = self.contexts_dir / context_file
            with open(context_path, 'r', encoding='utf-8') as f:
                context = yaml.safe_load(f)
            logger.info(f"Loaded context: {context_file}")
            return context
        except Exception as e:
            logger.error(f"Failed to load context {context_file}: {e}")
            return {}
    
    async def start_interview_session(self, 
                                    role: str, 
                                    level: str,
                                    candidate_name: Optional[str] = None,
                                    total_questions: int = 5) -> Dict:
        """
        TUTORIAL: Start a new interview session.
        
        This is like "opening a new interview folder" - it:
        1. Creates a unique session ID (timestamp-based)
        2. Initializes session state tracking
        3. Generates a personalized welcome message using AI
        4. Returns session info for the frontend
        
        Why async? Because we call AI (Ollama) to generate the welcome message.
        """
        # Generate unique session ID using timestamp
        session_id = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new session object
        session = InterviewSession(
            session_id=session_id,
            role=role,
            level=level,
            candidate_name=candidate_name,
            total_questions=total_questions
        )
        
        # Store in active sessions (in-memory storage)
        self.sessions[session_id] = session
        
        logger.info(f"Started interview session {session_id} for {role} {level}")
        
        # Generate fallback welcome message immediately (don't wait for AI)
        welcome_message = self._get_fallback_welcome_message(session)
        
        return {
            'session_id': session_id,
            'welcome_message': welcome_message,
            'session_info': {
                'role': role,
                'level': level,
                'total_questions': total_questions,
                'current_question': 0
            }
        }
    
    async def get_next_question(self, session_id: str) -> Optional[Dict]:
        """
        TUTORIAL: The core question selection and presentation logic.
        
        This is where the magic happens! It:
        1. Checks if interview is complete
        2. Selects appropriate question using RAG + strategy
        3. Uses AI to present the question naturally
        4. Updates session state
        5. Returns formatted question for frontend
        
        Strategy Pattern: Different question selection strategies based on
        interview progress (warmup -> progressive difficulty -> adaptive).
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Increment question counter first
        session.current_question_number += 1
        
        # Check if interview is complete AFTER incrementing
        logger.info(f"Question check for session {session_id}: current={session.current_question_number}, total={session.total_questions}")
        if session.current_question_number > session.total_questions:
            logger.info(f"Interview complete for session {session_id}: {session.current_question_number} > {session.total_questions}")
            return {
                'status': 'complete',
                'message': 'Interview completed! Thank you for your time.',
                'session_summary': await self._generate_session_summary(session)
            }
        
        # CORE LOGIC: Select appropriate question using intelligent strategy
        question = await self._select_question(session)
        
        if not question:
            logger.error(f"Failed to find question for session {session_id}")
            return {
                'status': 'error',
                'message': 'Unable to generate question. Please try again.'
            }
        
        # For performance: Skip AI presentation for RAG questions, only use AI for generated questions
        if hasattr(question, 'source') and question.source == 'rag':
            # RAG questions are already well-formatted, return directly
            question_presentation = question.content
        else:
            # Use AI to present AI-generated questions naturally
            question_presentation = await self._present_question(question, session)
        
        # Add to session history for tracking and future decision-making
        question_data = {
            'question_number': session.current_question_number,
            'question_id': question.question_id,
            'content': question.content,
            'topic': question.topic,
            'difficulty': question.difficulty,
            'asked_at': datetime.now().isoformat(),
            'expected_concepts': question.expected_concepts
        }
        
        session.questions_asked.append(question_data)
        
        return {
            'status': 'active',
            'question_number': session.current_question_number,
            'total_questions': session.total_questions,
            'question': question_presentation,
            'metadata': {
                'topic': question.topic,
                'difficulty': question.difficulty,
                'question_id': question.question_id
            }
        }
    
    async def _select_question(self, session: InterviewSession) -> Optional[Question]:
        """
        TUTORIAL: Intelligent question selection strategy.
        
        This implements the "Progressive Interview Strategy":
        
        Question 1: Easy fundamentals (warm-up)
        Question 2-3: Medium difficulty, explore different topics
        Question 4+: Adaptive difficulty based on performance
        
        Steps:
        1. Determine difficulty strategy based on progress
        2. Determine topic focus (cycling through role-relevant topics)
        3. Use RAG to find matching questions
        4. Filter out already-asked questions
        5. Select best match based on relevance score
        """
        
        # STRATEGY 1: Progressive Difficulty
        if session.current_question_number == 1:
            # First question: start with fundamentals (build confidence)
            difficulty = "easy"
            topic = "fundamentals"
        elif session.current_question_number <= 2:
            # Early questions: moderate difficulty
            difficulty = "medium"
            topic = await self._determine_topic_focus(session)
        else:
            # Later questions: adaptive difficulty (could analyze prev scores)
            difficulty = await self._determine_adaptive_difficulty(session)
            topic = await self._determine_topic_focus(session)
        
        # STRATEGY 2: Use RAG to find relevant questions
        rag_questions = await self.rag.search_questions(
            role=session.role,
            level=session.level,
            topic=topic,
            difficulty=difficulty,
            limit=10  # Get multiple options to choose from
        )
        
        if not rag_questions:
            logger.warning(f"No questions found for {session.role} {session.level}")
            # Fallback: broaden search criteria
            # Use at least the total questions needed, with some extra for variety
            fallback_limit = max(session.total_questions * 2, 5)
            rag_questions = await self.rag.search_questions(
                role=session.role,
                level=session.level,
                limit=fallback_limit
            )
        
        if not rag_questions:
            # Final fallback: generate question using AI when RAG fails
            return await self._generate_ai_question(session, topic, difficulty)
        
        # STRATEGY 3: Avoid repetition - filter out already asked questions
        asked_question_ids = {q.get('question_id') for q in session.questions_asked}
        asked_question_content = {q.get('content', '') for q in session.questions_asked}
        logger.info(f"Session {session.session_id}: Asked question IDs: {asked_question_ids}")
        logger.info(f"Session {session.session_id}: Found {len(rag_questions)} RAG questions")
        
        # Use content-based deduplication instead of ID-based (more reliable)
        available_questions = [
            q for q in rag_questions 
            if q.get('content', '') not in asked_question_content
        ]
        logger.info(f"Session {session.session_id}: Available questions after deduplication: {len(available_questions)}")
        
        # Debug: Log question content to check for duplicates
        for i, q in enumerate(available_questions[:3]):  # Log first 3 questions
            content_preview = q.get('content', 'N/A')[:50] + '...' if len(q.get('content', '')) > 50 else q.get('content', 'N/A')
            question_id = q['metadata'].get('question_id', 'N/A')
            logger.info(f"Available question {i+1}: ID={question_id}, Content='{content_preview}'")
        
        if not available_questions:
            # If all questions used, allow repetition (rare edge case)
            available_questions = rag_questions
        
        # STRATEGY 4: Select random question from available options for variety
        selected_rag = random.choice(available_questions)
        selected_id = selected_rag['metadata'].get('question_id', f"q_{session.current_question_number}")
        selected_content_preview = selected_rag.get('content', '')[:50] + '...' if len(selected_rag.get('content', '')) > 50 else selected_rag.get('content', '')
        logger.info(f"Selected question: ID={selected_id}, Content='{selected_content_preview}'")
        
        # Convert RAG result to our Question object
        question = Question(
            question_id=selected_rag['metadata'].get('question_id', f"q_{session.current_question_number}"),
            content=selected_rag['content'],
            role=session.role,
            level=session.level,
            topic=selected_rag['metadata'].get('topic', topic),
            difficulty=selected_rag['metadata'].get('difficulty', difficulty),
            expected_concepts=selected_rag['metadata'].get('expected_concepts', []),
            follow_up_prompts=selected_rag['metadata'].get('follow_up_prompts', [])
        )
        question.source = 'rag'  # Mark as RAG question for performance optimization
        
        return question
    
    async def _determine_topic_focus(self, session: InterviewSession) -> str:
        """
        TUTORIAL: Topic cycling strategy.
        
        Each role has a set of core topics that should be covered.
        This cycles through them to ensure comprehensive coverage.
        
        Example for Software Engineer:
        Q1: algorithms, Q2: data_structures, Q3: system_design, Q4: coding
        
        Example for Data Scientist:
        Q1: statistics, Q2: ml_modeling, Q3: data_analysis, Q4: experimentation
        """
        # Define core topics for each role (UPDATED: Added data_scientist)
        role_topics = {
            'software_engineer': ['algorithms', 'data_structures', 'system_design', 'coding'],
            'genai_engineer': ['machine_learning', 'nlp', 'transformers', 'llm_ops'],
            'devops': ['ci_cd', 'containerization', 'kubernetes', 'monitoring'],
            'data_engineer': ['data_pipelines', 'etl', 'databases', 'streaming'],
            'frontend': ['react', 'javascript', 'performance', 'accessibility'],
            'data_scientist': ['statistics', 'ml_modeling', 'data_analysis', 'experimentation', 'python_ml']
        }
        
        topics = role_topics.get(session.role, ['general', 'problem_solving'])
        
        # Cycle through topics based on question number (modulo for wraparound)
        topic_index = (session.current_question_number - 1) % len(topics)
        return topics[topic_index]
    
    async def _determine_adaptive_difficulty(self, session: InterviewSession) -> str:
        """
        TUTORIAL: Adaptive difficulty based on performance.
        
        In a real system, this would analyze scores from previous answers:
        - High scores → increase difficulty
        - Low scores → maintain or decrease difficulty
        
        For now, we use progressive difficulty as a starting point.
        """
        # TODO: In future, analyze previous answer scores here
        # For now, simple progressive difficulty
        if session.current_question_number <= 2:
            return "easy"
        elif session.current_question_number <= 4:
            return "medium"
        else:
            return "hard"
    
    async def _present_question(self, question: Question, session: InterviewSession) -> str:
        """
        TUTORIAL: AI-powered question presentation.
        
        Instead of just showing raw question text, this uses AI to:
        1. Add interviewer personality (from MCP context)
        2. Provide appropriate context and encouragement
        3. Format naturally like a human interviewer would
        
        This is where the MCP context really shines - it defines HOW
        the AI should behave as an interviewer.
        """
        # Build context for AI generation
        context = {
            'role': session.role,
            'level': session.level,
            'question_number': session.current_question_number,
            'total_questions': session.total_questions,
            'candidate_name': session.candidate_name,
            'question_content': question.content,
            'topic': question.topic,
            'difficulty': question.difficulty
        }
        
        # Create prompt using MCP context guidelines
        prompt = self._build_question_prompt(context)
        
        # Generate natural presentation using Ollama AI
        try:
            response = await self._call_ollama(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate question presentation: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to basic formatting if AI fails
            return self._format_basic_question(question, session)
    
    def _build_question_prompt(self, context: Dict) -> str:
        """
        TUTORIAL: Prompt engineering for question presentation.
        
        This takes the MCP context (interviewer personality) and combines
        it with the current question context to create a detailed prompt
        for the AI. The prompt tells the AI exactly how to behave.
        
        Key elements:
        1. Interviewer identity (from MCP context)
        2. Current interview context
        3. Specific question details
        4. Behavioral guidelines
        """
        # Extract interviewer personality from MCP context
        interviewer_identity = self.interviewer_context.get('role', {}).get('identity', '')
        questioning_style = self.interviewer_context.get('interview_methodology', {}).get('questioning_style', {})
        
        prompt = f"""
{interviewer_identity}

You are conducting a technical interview for a {context['role']} position at {context['level']} level.
This is question {context['question_number']} of {context['total_questions']}.

{f"The candidate's name is {context['candidate_name']}." if context['candidate_name'] else ""}

Present the following question in a professional, encouraging manner:

Question Topic: {context['topic']}
Difficulty: {context['difficulty']}

Question: {context['question_content']}

Guidelines:
- Be professional yet approachable
- Provide clear context if needed
- Encourage the candidate to think out loud
- Let them know they can ask for clarification

Format your response as a natural interview question presentation.
"""
        return prompt
    
    def _format_basic_question(self, question: Question, session: InterviewSession) -> str:
        """
        TUTORIAL: Fallback formatting when AI generation fails.
        
        Always have a fallback! If the AI service is down or fails,
        we still need to present questions in a reasonable format.
        """
        prefix = ""
        if session.candidate_name:
            prefix = f"{session.candidate_name}, "
        
        return f"""
{prefix}here's question {session.current_question_number} of {session.total_questions}.

**Question {session.current_question_number}**: {question.content}

Take your time to think through this. Feel free to ask for clarification if you need it, and please walk me through your thought process as you work through the answer.
"""
    
    async def _call_ollama(self, prompt: str) -> str:
        """
        TUTORIAL: Wrapper for calling Ollama API.
        
        This abstracts the HTTP API call to Ollama (our local AI).
        Key points:
        1. Uses async HTTP client for non-blocking calls
        2. Configures generation parameters (temperature, max_tokens)
        3. Handles errors gracefully
        4. Returns clean text response
        """
        payload = {
            "model": ollama_config.model_name,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.7,    # Balanced creativity vs consistency
                "num_predict": 150      # Shorter for faster response
            }
        }
        
        response = await self.ollama_client.post("/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    def _get_fallback_welcome_message(self, session: InterviewSession) -> str:
        """
        Generate a fallback welcome message immediately without AI delays.
        This ensures fast response times for starting interviews.
        """
        role_display = session.role.replace('_', ' ')
        return f"Welcome to your {role_display} technical interview! We'll be going through {session.total_questions} questions to assess your technical skills and experience. Feel free to think out loud and ask for clarification when needed. Let's begin!"

    async def _generate_welcome_message(self, session: InterviewSession) -> str:
        """
        TUTORIAL: AI-generated personalized welcome message.
        
        First impressions matter! This creates a warm, professional
        welcome that's tailored to the specific role and candidate.
        """
        context = {
            'role': session.role,
            'level': session.level,
            'candidate_name': session.candidate_name,
            'total_questions': session.total_questions
        }
        
        prompt = f"""
You are an expert technical interviewer starting an interview session.

Generate a warm, professional welcome message for a candidate interviewing for a {context['role']} position at {context['level']} level.

{f"The candidate's name is {context['candidate_name']}." if context['candidate_name'] else ""}

The interview will consist of {context['total_questions']} questions covering various technical aspects.

Make the candidate feel comfortable while setting professional expectations. Keep it concise but welcoming.
"""
        
        try:
            return await self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"Failed to generate welcome message: {e}")
            return self._get_fallback_welcome_message(session)
    
    async def _generate_session_summary(self, session: InterviewSession, backend_session: Dict = None) -> Dict:
        """
        TUTORIAL: Generate summary when interview completes.
        
        This provides useful metadata about the completed interview
        for analytics, reporting, and candidate feedback.
        """
        # Calculate average score from backend session data if available
        average_score = 0.0
        total_time = "Unknown"
        
        if backend_session and 'scores' in backend_session:
            scores = [s.get('overall_score', 0.0) for s in backend_session['scores']]
            average_score = sum(scores) / len(scores) if scores else 0.0
            
        if session.start_time:
            duration_minutes = (datetime.now() - session.start_time).total_seconds() / 60
            total_time = f"{duration_minutes:.1f} minutes"
        
        return {
            'session_id': session.session_id,
            'role': session.role,
            'level': session.level,
            'questions_completed': len(session.questions_asked),
            'total_questions': session.total_questions,
            'average_score': average_score,
            'total_time': total_time,
            'duration_minutes': (datetime.now() - session.start_time).total_seconds() / 60 if session.start_time else 0,
            'topics_covered': list(set(q.get('topic', 'unknown') for q in session.questions_asked))
        }
    
    async def _generate_ai_question(self, session: InterviewSession, topic: str, difficulty: str) -> Question:
        """
        Generate interview question using AI when RAG fails.
        Uses Ollama to create contextual, role-specific questions.
        """
        try:
            # Create prompt for question generation
            prompt = f"""Generate a {difficulty} difficulty technical interview question for a {session.level} {session.role.replace('_', ' ')}.

Topic focus: {topic}
Interview context: Question {session.current_question_number} of {session.total_questions}

Requirements:
- Make it relevant for {session.level} level
- Focus on {topic} if not 'general'
- Ensure it's appropriate for a {session.role.replace('_', ' ')} role
- Make it clear and specific
- No multiple choice - should require explanation

Generate only the question, nothing else:"""

            # Call Ollama to generate the question
            response = await self.ollama_client.post(
                "/api/generate",
                json={
                    "model": ollama_config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100,
                    }
                },
                timeout=300.0
            )
            
            if response.status_code == 200:
                response_data = response.json()
                question_content = response_data.get("response", "").strip()
                
                if question_content:
                    question = Question(
                        question_id=f"ai_generated_{session.session_id}_{session.current_question_number}",
                        content=question_content,
                        role=session.role,
                        level=session.level,
                        topic=topic,
                        difficulty=difficulty,
                        expected_concepts=[],
                        follow_up_prompts=[]
                    )
                    question.source = 'ai_generated'  # Mark as AI-generated question
                    return question
            
            logger.error(f"Failed to generate AI question: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error generating AI question: {e}")
        
        # If AI generation fails, return a simple generic question
        question = Question(
            question_id=f"generic_{session.current_question_number}",
            content=f"Tell me about your experience with {topic} in {session.role.replace('_', ' ')} roles.",
            role=session.role,
            level=session.level,
            topic=topic,
            difficulty=difficulty,
            expected_concepts=[],
            follow_up_prompts=[]
        )
        question.source = 'generic'  # Mark as generic fallback question
        return question

    # Utility methods for session management
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    async def end_session(self, session_id: str) -> Dict:
        """End an interview session and return summary."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        summary = await self._generate_session_summary(session)
        
        # Clean up - remove from active sessions
        del self.sessions[session_id]
        
        logger.info(f"Ended interview session {session_id}")
        return summary

# CLI interface for testing and learning
async def main():
    """
    TUTORIAL: CLI interface for testing the question asking system.
    
    This allows you to test the system from command line:
    python ask_question.py --role data_scientist --level senior --questions 3
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Question Asker for AI Interview Coach")
    parser.add_argument("--role", default="software_engineer", 
                       choices=["software_engineer", "genai_engineer", "devops", "data_engineer", "frontend", "data_scientist"],
                       help="Interview role")
    parser.add_argument("--level", default="mid", 
                       choices=["junior", "mid", "senior", "principal"],
                       help="Experience level")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions")
    parser.add_argument("--name", help="Candidate name")
    
    args = parser.parse_args()
    
    # Initialize logging so we can see what's happening
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components (RAG system and question asker)
    rag = RAGRetriever()
    await rag.load_documents()
    
    asker = QuestionAsker(rag)
    
    # Start interview session
    session_info = await asker.start_interview_session(
        role=args.role,
        level=args.level,
        candidate_name=args.name,
        total_questions=args.questions
    )
    
    print("=" * 50)
    print(session_info['welcome_message'])
    print("=" * 50)
    
    session_id = session_info['session_id']
    
    # Simulate asking questions
    for i in range(args.questions):
        print(f"\n--- Question {i + 1} ---")
        question_data = await asker.get_next_question(session_id)
        
        if question_data['status'] == 'active':
            print(question_data['question'])
            print(f"\nTopic: {question_data['metadata']['topic']}")
            print(f"Difficulty: {question_data['metadata']['difficulty']}")
            
            # Wait for user input (simulating candidate answering)
            input("\nPress Enter to continue to next question...")
        
        elif question_data['status'] == 'complete':
            print(question_data['message'])
            print("\nSession Summary:")
            summary = question_data['session_summary']
            for key, value in summary.items():
                print(f"- {key}: {value}")
            break
    
    # Clean up HTTP client
    await asker.ollama_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())