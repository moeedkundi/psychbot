"""
Feedback generation module for AI Interview Coach
Generates constructive, actionable feedback on interview answers using MCP context.

TUTORIAL: This script is the "AI Coach" - it analyzes candidate answers and provides
constructive feedback to help them improve. Think of it as a knowledgeable mentor who:

1. Identifies what the candidate did well (positive reinforcement)
2. Spots areas for improvement (growth opportunities)
3. Provides specific, actionable suggestions (next steps)
4. Tailors feedback to the role and experience level

Key Concepts:
- Uses MCP context to maintain consistent coaching personality
- Analyzes answers across multiple dimensions (technical accuracy, communication, etc.)
- Provides balanced feedback (strengths + improvements)
- Generates actionable suggestions with resources
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import ollama_config
import httpx

logger = logging.getLogger(__name__)

@dataclass
class AnswerAnalysis:
    """
    TUTORIAL: Structured analysis of a candidate's answer.
    
    This breaks down the answer evaluation into specific components
    that can be analyzed independently. Think of it as a "feedback rubric"
    that ensures we evaluate all important aspects consistently.
    """
    question_id: str
    question_content: str
    candidate_answer: str
    role: str
    level: str
    topic: str
    
    # Analysis results
    strengths: List[str]                    # What they did well
    improvement_areas: List[str]            # What needs work
    missing_concepts: List[str]             # Important concepts not mentioned
    communication_clarity: int              # 1-5 rating
    technical_accuracy: int                 # 1-5 rating
    depth_of_understanding: int             # 1-5 rating
    
    # Actionable suggestions
    specific_suggestions: List[str]         # Concrete next steps
    recommended_resources: List[str]        # Study materials/practices
    follow_up_questions: List[str]          # Questions to explore further

@dataclass
class FeedbackResult:
    """
    TUTORIAL: The final feedback output structure.
    
    This is what gets returned to the frontend and shown to the candidate.
    It's structured to be both human-readable and machine-processable.
    """
    question_number: int
    feedback_text: str              # Human-readable feedback
    analysis: AnswerAnalysis        # Structured analysis data
    overall_rating: float           # Overall performance score (1-5)
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'question_number': self.question_number,
            'feedback_text': self.feedback_text,
            'analysis': asdict(self.analysis),
            'overall_rating': self.overall_rating,
            'generated_at': self.generated_at.isoformat()
        }

class FeedbackGenerator:
    """
    TUTORIAL: The main feedback generation engine.
    
    This class is like having an expert technical coach who can:
    1. Analyze answers using structured criteria
    2. Generate natural, encouraging feedback
    3. Provide specific improvement suggestions
    4. Maintain consistency across all evaluations
    
    Architecture Pattern: This follows the "Analyzer-Generator" pattern:
    - Analyzer: Breaks down the answer into components
    - Generator: Creates natural language feedback from analysis
    """
    
    def __init__(self, contexts_dir: str = "contexts"):
        """
        TUTORIAL: Initialize the feedback generator.
        
        Loads the MCP feedback context which defines:
        - How to structure feedback (strengths → improvements → suggestions)
        - What tone to use (encouraging, specific, actionable)
        - What dimensions to evaluate (technical, communication, etc.)
        """
        self.contexts_dir = Path(contexts_dir)
        self.feedback_context = self._load_context("feedback_generator.context.yaml")
        
        # Initialize Ollama client for AI-powered feedback generation
        self.ollama_client = httpx.AsyncClient(
            base_url=ollama_config.base_url,
            timeout=ollama_config.timeout
        )
    
    def _load_context(self, context_file: str) -> Dict:
        """Load MCP context from YAML file."""
        try:
            context_path = self.contexts_dir / context_file
            with open(context_path, 'r', encoding='utf-8') as f:
                context = yaml.safe_load(f)
            logger.info(f"Loaded feedback context: {context_file}")
            return context
        except Exception as e:
            logger.error(f"Failed to load context {context_file}: {e}")
            return {}
    
    async def generate_feedback(self, 
                              question_content: str,
                              candidate_answer: str,
                              question_metadata: Dict,
                              session_context: Optional[Dict] = None) -> FeedbackResult:
        """
        TUTORIAL: Main feedback generation method.
        
        This is the core pipeline that transforms a raw answer into structured,
        actionable feedback. The process:
        
        1. ANALYZE: Break down the answer into components
        2. EVALUATE: Score each dimension (technical accuracy, clarity, etc.)
        3. GENERATE: Create natural language feedback using AI
        4. STRUCTURE: Package everything into a FeedbackResult
        
        Args:
            question_content: The original question asked
            candidate_answer: The candidate's response
            question_metadata: Question info (role, level, topic, difficulty)
            session_context: Optional context from the interview session
            
        Returns:
            FeedbackResult with structured feedback and analysis
        """
        
        # Step 1: Structured Analysis
        logger.info(f"Analyzing answer for {question_metadata.get('topic', 'unknown')} question")
        
        analysis = await self._analyze_answer(
            question_content=question_content,
            candidate_answer=candidate_answer,
            metadata=question_metadata,
            session_context=session_context
        )
        
        # Step 2: Generate Natural Language Feedback
        feedback_text = await self._generate_feedback_text(analysis)
        
        # Step 3: Calculate Overall Rating
        overall_rating = self._calculate_overall_rating(analysis)
        
        # Step 4: Package Results
        result = FeedbackResult(
            question_number=question_metadata.get('question_number', 1),
            feedback_text=feedback_text,
            analysis=analysis,
            overall_rating=overall_rating,
            generated_at=datetime.now()
        )
        
        logger.info(f"Generated feedback with overall rating: {overall_rating:.1f}/5.0")
        return result
    
    async def _analyze_answer(self, 
                            question_content: str,
                            candidate_answer: str,
                            metadata: Dict,
                            session_context: Optional[Dict] = None) -> AnswerAnalysis:
        """
        TUTORIAL: Structured answer analysis using AI.
        
        This is where we use AI to perform deep analysis of the candidate's answer.
        Instead of just generating free-form feedback, we ask the AI to analyze
        specific dimensions and provide structured output.
        
        The AI acts like a technical expert who systematically evaluates:
        - Technical accuracy (are the concepts correct?)
        - Communication clarity (is it well-explained?)
        - Completeness (are key concepts covered?)
        - Depth (how deep is the understanding?)
        """
        
        # Build analysis prompt using MCP context
        analysis_prompt = self._build_analysis_prompt(
            question_content, candidate_answer, metadata, session_context
        )
        
        try:
            # Get structured analysis from AI
            analysis_response = await self._call_ollama(analysis_prompt)
            
            # Parse AI response into structured data
            parsed_analysis = self._parse_analysis_response(
                analysis_response, question_content, candidate_answer, metadata
            )
            
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze answer: {e}")
            # Fallback to basic analysis
            return self._create_fallback_analysis(
                question_content, candidate_answer, metadata
            )
    
    def _build_analysis_prompt(self, 
                             question_content: str,
                             candidate_answer: str,
                             metadata: Dict,
                             session_context: Optional[Dict] = None) -> str:
        """
        TUTORIAL: Prompt engineering for structured answer analysis.
        
        This creates a detailed prompt that tells the AI exactly how to analyze
        the answer. The prompt includes:
        
        1. Context about the role and interview
        2. The specific question and answer
        3. Analysis framework (what to look for)
        4. Output format requirements
        
        Key Technique: We ask for JSON-structured output to make parsing easier.
        """
        
        # Extract role-specific expectations from metadata
        role = metadata.get('role', 'general')
        level = metadata.get('level', 'mid')
        topic = metadata.get('topic', 'technical')
        difficulty = metadata.get('difficulty', 'medium')
        
        # Get feedback philosophy from MCP context
        feedback_dimensions = self.feedback_context.get('feedback_dimensions', {})
        
        prompt = f"""
You are an expert technical coach analyzing a candidate's interview answer.

INTERVIEW CONTEXT:
- Role: {role}
- Level: {level}
- Topic: {topic}
- Difficulty: {difficulty}

QUESTION:
{question_content}

CANDIDATE'S ANSWER:
{candidate_answer}

Please analyze this answer across the following dimensions and provide a JSON response:

{{
  "strengths": ["list of specific things done well"],
  "improvement_areas": ["list of specific areas needing improvement"],
  "missing_concepts": ["important concepts not mentioned"],
  "communication_clarity": <rating 1-5>,
  "technical_accuracy": <rating 1-5>, 
  "depth_of_understanding": <rating 1-5>,
  "specific_suggestions": ["actionable improvement suggestions"],
  "recommended_resources": ["study materials or practices"],
  "follow_up_questions": ["questions to explore further"]
}}

ANALYSIS GUIDELINES:
- Be specific and constructive
- Focus on {level}-level expectations for {role} role
- Consider {topic} domain expertise
- Balance encouragement with honest assessment
- Provide actionable next steps

Respond with valid JSON only.
"""
        return prompt
    
    def _parse_analysis_response(self, 
                               response: str,
                               question_content: str,
                               candidate_answer: str,
                               metadata: Dict) -> AnswerAnalysis:
        """
        TUTORIAL: Parse AI response into structured AnswerAnalysis object.
        
        The AI returns JSON with the analysis. We parse this and create
        our structured AnswerAnalysis object. This includes error handling
        in case the AI doesn't return valid JSON.
        """
        try:
            # Try to parse JSON response
            analysis_data = json.loads(response.strip())
            
            return AnswerAnalysis(
                question_id=metadata.get('question_id', 'unknown'),
                question_content=question_content,
                candidate_answer=candidate_answer,
                role=metadata.get('role', 'general'),
                level=metadata.get('level', 'mid'),
                topic=metadata.get('topic', 'technical'),
                
                # Extract analysis results with defaults
                strengths=analysis_data.get('strengths', []),
                improvement_areas=analysis_data.get('improvement_areas', []),
                missing_concepts=analysis_data.get('missing_concepts', []),
                communication_clarity=analysis_data.get('communication_clarity', 3),
                technical_accuracy=analysis_data.get('technical_accuracy', 3),
                depth_of_understanding=analysis_data.get('depth_of_understanding', 3),
                
                # Extract suggestions
                specific_suggestions=analysis_data.get('specific_suggestions', []),
                recommended_resources=analysis_data.get('recommended_resources', []),
                follow_up_questions=analysis_data.get('follow_up_questions', [])
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback to basic analysis
            return self._create_fallback_analysis(question_content, candidate_answer, metadata)
    
    def _create_fallback_analysis(self, 
                                question_content: str,
                                candidate_answer: str,
                                metadata: Dict) -> AnswerAnalysis:
        """
        TUTORIAL: Fallback analysis when AI parsing fails.
        
        Always have a backup plan! If the AI doesn't return parseable JSON,
        we create a basic analysis with default values. This ensures the
        system keeps working even if the AI has issues.
        """
        return AnswerAnalysis(
            question_id=metadata.get('question_id', 'unknown'),
            question_content=question_content,
            candidate_answer=candidate_answer,
            role=metadata.get('role', 'general'),
            level=metadata.get('level', 'mid'),
            topic=metadata.get('topic', 'technical'),
            
            # Default analysis
            strengths=["Provided a response to the question"],
            improvement_areas=["Could provide more detailed explanation"],
            missing_concepts=[],
            communication_clarity=3,
            technical_accuracy=3,
            depth_of_understanding=3,
            
            # Default suggestions
            specific_suggestions=["Practice explaining technical concepts clearly"],
            recommended_resources=["Review fundamentals of the topic"],
            follow_up_questions=["What aspects would you like to explore further?"]
        )
    
    async def _generate_feedback_text(self, analysis: AnswerAnalysis) -> str:
        """
        TUTORIAL: Generate natural language feedback from structured analysis.
        
        This takes our structured analysis and converts it into natural,
        human-friendly feedback text. The AI uses the MCP context to maintain
        a consistent coaching tone and structure.
        
        The feedback follows a proven structure:
        1. Start with strengths (positive reinforcement)
        2. Identify improvement areas (growth opportunities)  
        3. Provide specific suggestions (actionable next steps)
        4. End with encouragement (growth mindset)
        """
        
        # Build feedback generation prompt
        feedback_prompt = self._build_feedback_prompt(analysis)
        
        try:
            feedback_text = await self._call_ollama(feedback_prompt)
            return feedback_text
            
        except Exception as e:
            logger.error(f"Failed to generate feedback text: {e}")
            # Fallback to template-based feedback
            return self._generate_template_feedback(analysis)
    
    def _build_feedback_prompt(self, analysis: AnswerAnalysis) -> str:
        """
        TUTORIAL: Prompt for generating natural feedback text.
        
        This prompt tells the AI how to convert our structured analysis
        into natural, encouraging feedback. It uses the MCP context to
        maintain consistent tone and structure.
        """
        
        # Get feedback tone guidelines from MCP context
        feedback_structure = self.feedback_context.get('feedback_structure', {})
        feedback_tone = self.feedback_context.get('feedback_tone', {})
        
        prompt = f"""
You are an expert technical coach providing constructive feedback to a {analysis.level} {analysis.role} candidate.

QUESTION TOPIC: {analysis.topic}

ANALYSIS RESULTS:
Strengths: {', '.join(analysis.strengths)}
Improvement Areas: {', '.join(analysis.improvement_areas)}
Missing Concepts: {', '.join(analysis.missing_concepts)}

RATINGS:
- Communication Clarity: {analysis.communication_clarity}/5
- Technical Accuracy: {analysis.technical_accuracy}/5  
- Depth of Understanding: {analysis.depth_of_understanding}/5

SUGGESTIONS: {', '.join(analysis.specific_suggestions)}
RESOURCES: {', '.join(analysis.recommended_resources)}

Generate encouraging, constructive feedback following this structure:

## 🎯 Strengths
[Highlight specific things done well with examples]

## 🔧 Areas for Improvement  
[Identify specific areas to develop with context]

## 💡 Specific Suggestions
[Provide actionable next steps]

## 📚 Resources to Explore
[Recommend study materials if applicable]

Use an encouraging, professional tone. Be specific with examples. Focus on growth and learning opportunities.
"""
        return prompt
    
    def _generate_template_feedback(self, analysis: AnswerAnalysis) -> str:
        """
        TUTORIAL: Template-based feedback fallback.
        
        If AI generation fails, we use a template to ensure consistent
        feedback structure. This is less personalized but still useful.
        """
        
        strengths_text = "\n".join([f"- {strength}" for strength in analysis.strengths])
        improvements_text = "\n".join([f"- {area}" for area in analysis.improvement_areas])
        suggestions_text = "\n".join([f"- {suggestion}" for suggestion in analysis.specific_suggestions])
        
        feedback = f"""
## 🎯 Strengths
{strengths_text if strengths_text else "- Provided a complete response to the question"}

## 🔧 Areas for Improvement
{improvements_text if improvements_text else "- Consider providing more detailed explanations"}

## 💡 Specific Suggestions
{suggestions_text if suggestions_text else "- Practice explaining technical concepts step by step"}

Keep up the good work and focus on these areas for continued improvement!
"""
        return feedback.strip()
    
    def _calculate_overall_rating(self, analysis: AnswerAnalysis) -> float:
        """
        TUTORIAL: Calculate overall performance rating.
        
        This combines the individual dimension scores into an overall rating.
        We use weighted averages to emphasize the most important aspects.
        
        Weights (based on feedback context):
        - Technical Accuracy: 40% (most important)
        - Communication Clarity: 35% (very important for interviews)
        - Depth of Understanding: 25% (shows mastery level)
        """
        
        # Define weights for different dimensions
        weights = {
            'technical_accuracy': 0.40,
            'communication_clarity': 0.35, 
            'depth_of_understanding': 0.25
        }
        
        # Calculate weighted average
        weighted_sum = (
            analysis.technical_accuracy * weights['technical_accuracy'] +
            analysis.communication_clarity * weights['communication_clarity'] +
            analysis.depth_of_understanding * weights['depth_of_understanding']
        )
        
        # Round to 1 decimal place
        return round(weighted_sum, 1)
    
    async def _call_ollama(self, prompt: str) -> str:
        """
        TUTORIAL: Call Ollama API for text generation.
        
        This is our interface to the local Mistral AI model.
        We configure it for feedback generation with appropriate parameters.
        """
        payload = {
            "model": ollama_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,    # Balanced creativity for natural feedback
                "num_predict": 300,    # Shorter feedback for faster response
                "top_p": 0.9          # Good diversity in language
            }
        }
        
        response = await self.ollama_client.post("/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    async def generate_batch_feedback(self, 
                                    answer_data_list: List[Dict]) -> List[FeedbackResult]:
        """
        TUTORIAL: Generate feedback for multiple answers at once.
        
        This is useful for processing an entire interview session or
        when you need to analyze multiple answers efficiently.
        """
        feedback_results = []
        
        for answer_data in answer_data_list:
            try:
                feedback = await self.generate_feedback(
                    question_content=answer_data['question_content'],
                    candidate_answer=answer_data['candidate_answer'],
                    question_metadata=answer_data['metadata'],
                    session_context=answer_data.get('session_context')
                )
                feedback_results.append(feedback)
                
            except Exception as e:
                logger.error(f"Failed to generate feedback for question {answer_data.get('question_id')}: {e}")
                # Continue with other questions even if one fails
                continue
        
        return feedback_results
    
    def get_feedback_summary(self, feedback_results: List[FeedbackResult]) -> Dict:
        """
        TUTORIAL: Generate summary across multiple feedback results.
        
        This analyzes patterns across all questions to provide insights like:
        - Overall performance trends
        - Strengths and improvement areas across topics
        - Recommended focus areas for development
        """
        if not feedback_results:
            return {'error': 'No feedback results to summarize'}
        
        # Calculate aggregate metrics
        avg_rating = sum(f.overall_rating for f in feedback_results) / len(feedback_results)
        
        # Collect all strengths and improvement areas
        all_strengths = []
        all_improvements = []
        topic_ratings = {}
        
        for feedback in feedback_results:
            all_strengths.extend(feedback.analysis.strengths)
            all_improvements.extend(feedback.analysis.improvement_areas)
            
            # Track ratings by topic
            topic = feedback.analysis.topic
            if topic not in topic_ratings:
                topic_ratings[topic] = []
            topic_ratings[topic].append(feedback.overall_rating)
        
        # Calculate topic averages
        topic_averages = {
            topic: sum(ratings) / len(ratings) 
            for topic, ratings in topic_ratings.items()
        }
        
        return {
            'overall_rating': round(avg_rating, 1),
            'total_questions': len(feedback_results),
            'topic_performance': topic_averages,
            'common_strengths': self._find_common_themes(all_strengths),
            'common_improvement_areas': self._find_common_themes(all_improvements),
            'strongest_topic': max(topic_averages.items(), key=lambda x: x[1])[0] if topic_averages else None,
            'focus_topic': min(topic_averages.items(), key=lambda x: x[1])[0] if topic_averages else None
        }
    
    def _find_common_themes(self, items: List[str], threshold: int = 2) -> List[str]:
        """Find themes that appear multiple times across feedback."""
        # Simple keyword-based theme detection
        word_counts = {}
        for item in items:
            words = item.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return words that appear multiple times
        common_themes = [word for word, count in word_counts.items() if count >= threshold]
        return common_themes[:5]  # Top 5 themes
    
    async def close(self):
        """Clean up resources."""
        await self.ollama_client.aclose()

# CLI interface for testing
async def main():
    """
    TUTORIAL: CLI interface for testing feedback generation.
    
    This allows you to test the feedback system with sample questions and answers:
    python generate_feedback.py --question "Explain algorithms" --answer "Algorithms are step-by-step procedures"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Feedback Generator for AI Interview Coach")
    parser.add_argument("--question", required=True, help="Question content")
    parser.add_argument("--answer", required=True, help="Candidate's answer")
    parser.add_argument("--role", default="software_engineer", help="Interview role")
    parser.add_argument("--level", default="mid", help="Experience level")
    parser.add_argument("--topic", default="technical", help="Question topic")
    parser.add_argument("--difficulty", default="medium", help="Question difficulty")
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Create feedback generator
    generator = FeedbackGenerator()
    
    # Prepare question metadata
    metadata = {
        'question_id': 'test_question_1',
        'question_number': 1,
        'role': args.role,
        'level': args.level,
        'topic': args.topic,
        'difficulty': args.difficulty
    }
    
    try:
        # Generate feedback
        print("Generating feedback...")
        feedback = await generator.generate_feedback(
            question_content=args.question,
            candidate_answer=args.answer,
            question_metadata=metadata
        )
        
        print("\n" + "="*50)
        print("FEEDBACK RESULT")
        print("="*50)
        print(f"Overall Rating: {feedback.overall_rating}/5.0")
        print(f"Question: {args.question}")
        print(f"Answer: {args.answer}")
        print("\n" + feedback.feedback_text)
        
        print("\n" + "="*50)
        print("DETAILED ANALYSIS")
        print("="*50)
        analysis = feedback.analysis
        print(f"Technical Accuracy: {analysis.technical_accuracy}/5")
        print(f"Communication Clarity: {analysis.communication_clarity}/5")
        print(f"Depth of Understanding: {analysis.depth_of_understanding}/5")
        print(f"Strengths: {analysis.strengths}")
        print(f"Improvement Areas: {analysis.improvement_areas}")
        print(f"Suggestions: {analysis.specific_suggestions}")
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main())