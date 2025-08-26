"""
FastAPI backend server for AI Interview Coach
Provides REST API endpoints for the interview chatbot functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import uuid

# Import our core modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.ask_question import QuestionAsker, InterviewSession
from scripts.generate_feedback import FeedbackGenerator, FeedbackResult
from scripts.score_answer import AnswerScorer, ScoreResult
from scripts.generate_report import ReportGenerator, ReportConfiguration, generate_candidate_report
from scripts.rag_retriever import RAGRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="AI Interview Coach API",
    description="REST API for conducting AI-powered technical interviews",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components - initialized on startup
question_asker: Optional[QuestionAsker] = None
feedback_generator: Optional[FeedbackGenerator] = None
answer_scorer: Optional[AnswerScorer] = None
report_generator: Optional[ReportGenerator] = None
rag_retriever: Optional[RAGRetriever] = None
comprehensive_evaluator = None

# In-memory storage for active sessions (in production, use Redis or database)
active_sessions: Dict[str, Dict] = {}
session_results: Dict[str, Dict] = {}

# Pydantic models for request/response validation
class StartInterviewRequest(BaseModel):
    role: str = Field(..., description="Interview role (e.g., software_engineer, data_scientist)")
    level: str = Field(..., description="Experience level (junior, mid, senior, principal)")
    candidate_name: Optional[str] = Field(None, description="Optional candidate name")
    total_questions: int = Field(5, description="Number of questions to ask", ge=1, le=50)

class StartInterviewResponse(BaseModel):
    session_id: str
    welcome_message: str
    session_info: Dict[str, Any]

class GetQuestionResponse(BaseModel):
    status: str
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    question: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    session_summary: Optional[Dict[str, Any]] = None

class SubmitAnswerRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    answer: str = Field(..., description="Candidate's answer to the current question")

class SubmitAnswerResponse(BaseModel):
    feedback: Dict[str, Any]
    score: Dict[str, Any]
    next_question: Optional[Dict[str, Any]] = None

class GenerateReportRequest(BaseModel):
    session_id: str = Field(..., description="Completed session ID")
    report_type: str = Field("candidate", description="Type of report to generate")
    format: str = Field("markdown", description="Output format")
    include_charts: bool = Field(True, description="Include performance charts")

class GenerateReportResponse(BaseModel):
    report_file: str
    download_url: str
    metadata: Dict[str, Any]

# Startup event to initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize all AI components on server startup."""
    global question_asker, feedback_generator, answer_scorer, report_generator, rag_retriever, comprehensive_evaluator
    
    logger.info("Initializing AI Interview Coach components...")
    
    try:
        # Initialize RAG retriever (skip document loading for now to avoid segfault)
        try:
            rag_retriever = RAGRetriever(
                data_dir="data/vector_db",
                docs_dir="docs"
            )
            # Load documents into RAG system
            rag_retriever.load_documents()
            logger.info("RAG documents loaded successfully")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            rag_retriever = None
        
        # Initialize comprehensive evaluator (single instance for all requests)
        from scripts.comprehensive_evaluator import get_comprehensive_evaluator
        comprehensive_evaluator = await get_comprehensive_evaluator()
        logger.info("Comprehensive evaluator initialized successfully")
        
        # Initialize other components
        question_asker = QuestionAsker(rag_retriever)
        feedback_generator = FeedbackGenerator()
        answer_scorer = AnswerScorer()
        report_generator = ReportGenerator()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown."""
    logger.info("Shutting down AI Interview Coach...")
    
    # Close any async clients
    if comprehensive_evaluator:
        await comprehensive_evaluator.close()
    if feedback_generator:
        await feedback_generator.close()
    if answer_scorer:
        await answer_scorer.close()
    if report_generator:
        await report_generator.close()
    if question_asker:
        await question_asker.ollama_client.aclose()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "question_asker": question_asker is not None,
            "feedback_generator": feedback_generator is not None,
            "answer_scorer": answer_scorer is not None,
            "report_generator": report_generator is not None,
            "rag_retriever": rag_retriever is not None,
            "comprehensive_evaluator": comprehensive_evaluator is not None
        }
    }

# Interview endpoints
@app.post("/api/interview/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    """Start a new interview session."""
    try:
        if not question_asker:
            raise HTTPException(status_code=500, detail="Question asker not initialized")
        
        # Start interview session
        session_data = await question_asker.start_interview_session(
            role=request.role,
            level=request.level,
            candidate_name=request.candidate_name,
            total_questions=request.total_questions
        )
        
        session_id = session_data['session_id']
        
        # Store session in memory
        active_sessions[session_id] = {
            'session_data': session_data,
            'questions_asked': [],
            'answers_submitted': [],
            'scores': [],
            'feedback': [],
            'current_question': None,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Started interview session {session_id} for {request.role} {request.level}")
        
        return StartInterviewResponse(
            session_id=session_id,
            welcome_message=session_data['welcome_message'],
            session_info=session_data['session_info']
        )
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview/{session_id}/question", response_model=GetQuestionResponse)
async def get_next_question(session_id: str):
    """Get the next question for an interview session."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not question_asker:
            raise HTTPException(status_code=500, detail="Question asker not initialized")
        
        # Get next question
        question_data = await question_asker.get_next_question(session_id)
        
        if question_data['status'] == 'active':
            # Store current question in session
            active_sessions[session_id]['current_question'] = question_data
            
        elif question_data['status'] == 'complete':
            # Move session to completed
            session_results[session_id] = active_sessions.pop(session_id)
        
        return GetQuestionResponse(**question_data)
        
    except Exception as e:
        logger.error(f"Error getting question for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview/answer", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """Submit an answer and get feedback and scoring."""
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        current_question = session.get('current_question')
        
        if not current_question:
            raise HTTPException(status_code=400, detail="No active question for this session")
        
        # Prepare question metadata
        question_metadata = {
            'question_id': current_question['metadata']['question_id'],
            'question_number': current_question['question_number'],
            'role': session['session_data']['session_info']['role'],
            'level': session['session_data']['session_info']['level'],
            'topic': current_question['metadata']['topic'],
            'difficulty': current_question['metadata']['difficulty']
        }
        
        # Use global comprehensive evaluator instance (no per-request initialization)
        if not comprehensive_evaluator:
            raise HTTPException(status_code=500, detail="Comprehensive evaluator not initialized")
        
        # Single comprehensive evaluation call
        evaluation_result = await comprehensive_evaluator.evaluate_answer(
            question_content=current_question['question'],
            candidate_answer=request.answer,
            question_metadata=question_metadata
        )
        
        # Extract feedback and score from comprehensive evaluation
        feedback_result = {
            'overall_rating': evaluation_result['overall_score'],
            'feedback': evaluation_result['feedback'],
            'detailed_scores': evaluation_result['scores'],
            'follow_up_question': evaluation_result.get('follow_up')
        }
        
        score_result = {
            'overall_score': evaluation_result['overall_score'],
            'category_scores': evaluation_result['scores'],
            'rationale': evaluation_result.get('rationale', {})
        }
        
        # Store results in session
        session['questions_asked'].append(current_question)
        session['answers_submitted'].append({
            'question_id': current_question['metadata']['question_id'],
            'question_number': current_question['question_number'],
            'answer': request.answer,
            'submitted_at': datetime.now().isoformat()
        })
        session['scores'].append(score_result)
        session['feedback'].append(feedback_result)
        session['current_question'] = None  # Clear current question
        
        # Get next question
        next_question_data = await question_asker.get_next_question(session_id)
        next_question = None
        
        if next_question_data['status'] == 'active':
            session['current_question'] = next_question_data
            next_question = next_question_data
        elif next_question_data['status'] == 'complete':
            # Calculate proper session summary from backend data
            completed_session = active_sessions[session_id]
            scores = [s.get('overall_score', 0.0) for s in completed_session['scores']]
            average_score = sum(scores) / len(scores) if scores else 0.0
            
            # Debug logging
            logger.info(f"Session {session_id} completion - Raw scores: {completed_session['scores']}")
            logger.info(f"Session {session_id} completion - Extracted scores: {scores}")
            logger.info(f"Session {session_id} completion - Calculated average: {average_score}")
            
            # Calculate total time (with fallback)
            try:
                session_start_str = completed_session['session_data']['session_info'].get('start_time')
                if session_start_str:
                    session_start = datetime.fromisoformat(session_start_str)
                    duration_minutes = (datetime.now() - session_start).total_seconds() / 60
                    total_time = f"{duration_minutes:.1f} minutes"
                else:
                    total_time = "Unknown"
                    duration_minutes = 0
            except Exception:
                total_time = "Unknown" 
                duration_minutes = 0
            
            # Create proper session summary
            session_summary = {
                'session_id': session_id,
                'questions_completed': len(completed_session['answers_submitted']),
                'total_questions': completed_session['session_data']['session_info']['total_questions'],
                'average_score': average_score,
                'total_time': total_time,
                'duration_minutes': duration_minutes
            }
            
            # Update the next_question data with correct session summary
            next_question_data['session_summary'] = session_summary
            
            # Move to completed sessions
            session_results[session_id] = active_sessions.pop(session_id)
            next_question = next_question_data
        
        logger.info(f"Processed answer for session {session_id}, question {current_question['question_number']}")
        
        return SubmitAnswerResponse(
            feedback=feedback_result,
            score=score_result,
            next_question=next_question
        )
        
    except Exception as e:
        logger.error(f"Error processing answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current status of an interview session."""
    try:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            return {
                "status": "active",
                "session_id": session_id,
                "questions_completed": len(session['answers_submitted']),
                "total_questions": session['session_data']['session_info']['total_questions'],
                "has_current_question": session['current_question'] is not None
            }
        elif session_id in session_results:
            session = session_results[session_id]
            return {
                "status": "completed",
                "session_id": session_id,
                "questions_completed": len(session['answers_submitted']),
                "total_questions": session['session_data']['session_info']['total_questions'],
                "completed_at": session.get('completed_at')
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Report generation endpoints
@app.post("/api/report/generate", response_model=GenerateReportResponse)
async def generate_report(request: GenerateReportRequest):
    """Generate a comprehensive interview report."""
    try:
        session_id = request.session_id
        
        if session_id not in session_results:
            raise HTTPException(status_code=404, detail="Completed session not found")
        
        session = session_results[session_id]
        
        # Create report configuration
        config = ReportConfiguration(
            report_type=request.report_type,
            format=request.format,
            include_detailed_scores=True,
            include_improvement_suggestions=True,
            include_charts=request.include_charts,
            include_raw_answers=False
        )
        
        # Generate report with dictionary data (report generator will handle conversion)
        report_result = await report_generator.generate_interview_report(
            session_data=session['session_data']['session_info'],
            scores=session['scores'],
            feedback=session['feedback'],
            config=config
        )
        
        logger.info(f"Generated {request.report_type} report for session {session_id}")
        
        return GenerateReportResponse(
            report_file=report_result['report_file'],
            download_url=f"/api/report/download/{session_id}",
            metadata=report_result['metadata']
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/download/{session_id}")
async def download_report(session_id: str):
    """Download the generated report file."""
    # In a real implementation, you'd serve the actual file
    # For now, return a placeholder
    return {"message": f"Report download for session {session_id} - file serving not implemented in demo"}

# Analytics and monitoring endpoints
@app.get("/api/analytics/sessions")
async def get_session_analytics():
    """Get analytics about interview sessions."""
    try:
        total_active = len(active_sessions)
        total_completed = len(session_results)
        
        # Calculate completion rates by role
        role_stats = {}
        for session in session_results.values():
            role = session['session_data']['session_info']['role']
            if role not in role_stats:
                role_stats[role] = {'completed': 0, 'avg_score': 0}
            
            role_stats[role]['completed'] += 1
            
            # Calculate average score
            if session['scores']:
                avg_score = sum(score.overall_score for score in session['scores']) / len(session['scores'])
                role_stats[role]['avg_score'] = avg_score
        
        return {
            "total_active_sessions": total_active,
            "total_completed_sessions": total_completed,
            "role_statistics": role_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/stats")
async def get_rag_statistics():
    """Get statistics about the RAG knowledge base."""
    try:
        if not rag_retriever:
            raise HTTPException(status_code=500, detail="RAG retriever not initialized")
        
        stats = rag_retriever.get_collection_stats()
        return {
            "knowledge_base_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development and testing endpoints
@app.post("/api/dev/reload-documents")
async def reload_documents():
    """Reload RAG documents (development endpoint)."""
    try:
        if not rag_retriever:
            raise HTTPException(status_code=500, detail="RAG retriever not initialized")
        
        rag_retriever.load_documents(force_reload=True)
        stats = rag_retriever.get_collection_stats()
        
        return {
            "message": "Documents reloaded successfully",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )