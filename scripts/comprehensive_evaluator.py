"""
Answer Evaluation System
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.config import ollama_config

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    
    def __init__(self, contexts_dir: str = "contexts"):
        self.contexts_dir = Path(contexts_dir)
        
        # Initialize Ollama client with explicit long timeout
        self.ollama_client = httpx.AsyncClient(
            base_url=ollama_config.base_url,
            timeout=httpx.Timeout(connect=60.0, read=300.0, write=300.0, pool=300.0)
        )
        
        logger.info(f"Comprehensive evaluator using {ollama_config.model_name} at {ollama_config.base_url}")
        
        # Load evaluation context
        self.evaluation_context = self._load_context("comprehensive_evaluator.context.yaml")
        
        logger.info("Comprehensive evaluator initialized")
    
    def _load_context(self, context_file: str) -> Dict:
        """Load evaluation context from YAML file."""
        context_path = self.contexts_dir / context_file
        
        if not context_path.exists():
            logger.warning(f"Context file not found: {context_path}")
            return self._get_default_context()
        
        try:
            import yaml
            with open(context_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load context {context_path}: {e}")
            return self._get_default_context()
    
    def _get_default_context(self) -> Dict:
        """Default evaluation context if file not found."""
        return {
            "role": "technical_interviewer",
            "evaluation_criteria": {
                "technical_accuracy": "Correctness of technical concepts and facts",
                "clarity_communication": "How clearly the answer is expressed",
                "completeness_depth": "How thorough and detailed the answer is",
                "problem_solving": "Evidence of analytical thinking and problem-solving approach",
                "practical_application": "Demonstrates real-world understanding and experience"
            },
            "scoring_scale": {
                "1": "Poor - Major gaps or incorrect information",
                "2": "Below Average - Some correct elements but significant issues",
                "3": "Satisfactory - Adequate answer with minor issues",
                "4": "Good - Strong answer with good understanding",
                "5": "Excellent - Comprehensive, accurate, and insightful answer"
            }
        }
    
    async def evaluate_answer(self, 
                            question_content: str, 
                            candidate_answer: str, 
                            question_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive single-call answer evaluation.
        
        Returns complete evaluation including scores, feedback, and rationale
        in one optimized AI call instead of 7 separate calls.
        """
        try:
            # Build comprehensive evaluation prompt
            prompt = self._build_evaluation_prompt(
                question_content, candidate_answer, question_metadata
            )
            
            # Single AI call for complete evaluation
            start_time = asyncio.get_event_loop().time()
            evaluation_text = await self._call_ollama(prompt)
            duration = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Comprehensive evaluation completed in {duration:.2f}s")
            
            # Parse structured evaluation response
            evaluation = self._parse_evaluation_response(evaluation_text)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate answer: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_fallback_evaluation(question_metadata)
    
    def _build_evaluation_prompt(self, question_content: str, candidate_answer: str, 
                               question_metadata: Dict[str, Any]) -> str:
        """Build optimized prompt for comprehensive evaluation."""
        
        role = question_metadata.get('role', 'technical_professional')
        level = question_metadata.get('level', 'mid')
        topic = question_metadata.get('topic', 'technical')
        
        prompt = f"""Evaluate this {level} {role} candidate's answer. Be concise.

Q: {question_content}
A: {candidate_answer}

Rate 1-5 and give brief feedback:

SCORES:
- Technical Accuracy: [score]
- Clarity: [score]
- Completeness: [score]
- Problem Solving: [score]
- Practical Application: [score]

OVERALL SCORE: [average]/5.0

FEEDBACK: [1-2 sentences with key strengths/improvements]"""

        return prompt
    
    def _parse_evaluation_response(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse AI response into structured evaluation data."""
        
        try:
            lines = evaluation_text.strip().split('\n')
            
            # Initialize result structure
            evaluation = {
                'scores': {
                    'technical_accuracy': 3.0,
                    'clarity_communication': 3.0,
                    'completeness_depth': 3.0,
                    'problem_solving': 3.0,
                    'practical_application': 3.0
                },
                'overall_score': 3.0,
                'feedback': "Answer evaluated. Please refer to individual scores for details.",
                'follow_up': None,
                'rationale': {}
            }
            
            current_section = None
            feedback_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections (handle both SCORES: and SCORES (formats)
                if 'SCORES' in line.upper():
                    current_section = 'scores'
                    continue
                elif line.startswith('OVERALL SCORE'):
                    # Extract overall score
                    score_match = self._extract_score_from_line(line)
                    if score_match:
                        evaluation['overall_score'] = score_match
                    current_section = 'overall'
                    continue
                elif 'FEEDBACK' in line.upper():
                    current_section = 'feedback'
                    # Extract feedback from same line if it exists
                    if ':' in line:
                        feedback_part = line.split(':', 1)[1].strip()
                        if feedback_part:
                            feedback_lines.append(feedback_part)
                    continue
                elif line.startswith('FOLLOW-UP'):
                    current_section = 'follow_up'
                    continue
                
                # Process content based on current section
                if current_section == 'scores':
                    self._parse_score_line(line, evaluation)
                elif current_section == 'feedback':
                    feedback_lines.append(line)
                elif current_section == 'follow_up':
                    if line.lower() != 'none':
                        evaluation['follow_up'] = line
                
                # Handle alternative formats - look for bullet points with scores anywhere
                if line.startswith('*') or line.startswith('-'):
                    self._parse_score_line(line, evaluation)
                
                # Look for feedback text after "FEEDBACK" anywhere
                if current_section is None and any(keyword in line.lower() for keyword in ['feedback', 'summary', 'evaluation']):
                    if ':' in line:
                        feedback_part = line.split(':', 1)[1].strip()
                        if feedback_part:
                            feedback_lines.append(feedback_part)
                
                # Capture feedback content when in feedback section
                if current_section == 'feedback' and ':' in line:
                    feedback_part = line.split(':', 1)[1].strip()
                    if feedback_part:
                        feedback_lines.append(feedback_part)
            
            # Combine feedback lines
            if feedback_lines:
                evaluation['feedback'] = ' '.join(feedback_lines)
            
            # Calculate overall score if not parsed
            if evaluation['overall_score'] == 3.0:
                scores = evaluation['scores'].values()
                evaluation['overall_score'] = round(sum(scores) / len(scores), 1)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return self._get_fallback_evaluation({})
    
    def _extract_score_from_line(self, line: str) -> Optional[float]:
        """Extract numeric score from a line of text."""
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)', line)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                pass
        return None
    
    def _parse_score_line(self, line: str, evaluation: Dict[str, Any]) -> None:
        """Parse individual score line and update evaluation."""
        line_lower = line.lower()
        
        score = self._extract_score_from_line(line)
        if not score:
            return
        
        # Map line content to score categories
        if 'technical' in line_lower and 'accuracy' in line_lower:
            evaluation['scores']['technical_accuracy'] = score
            evaluation['rationale']['technical_accuracy'] = line
        elif 'clarity' in line_lower or 'communication' in line_lower:
            evaluation['scores']['clarity_communication'] = score
            evaluation['rationale']['clarity_communication'] = line
        elif 'completeness' in line_lower or 'depth' in line_lower:
            evaluation['scores']['completeness_depth'] = score
            evaluation['rationale']['completeness_depth'] = line
        elif 'problem' in line_lower and 'solving' in line_lower:
            evaluation['scores']['problem_solving'] = score
            evaluation['rationale']['problem_solving'] = line
        elif 'practical' in line_lower or 'application' in line_lower:
            evaluation['scores']['practical_application'] = score
            evaluation['rationale']['practical_application'] = line
    
    def _get_fallback_evaluation(self, question_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback evaluation when AI processing fails."""
        return {
            'scores': {
                'technical_accuracy': 3.0,
                'clarity_communication': 3.0,
                'completeness_depth': 3.0,
                'problem_solving': 3.0,
                'practical_application': 3.0
            },
            'overall_score': 3.0,
            'feedback': "Your answer has been received and noted. Thank you for your response.",
            'follow_up': None,
            'rationale': {
                'technical_accuracy': "Standard evaluation",
                'clarity_communication': "Standard evaluation",
                'completeness_depth': "Standard evaluation",
                'problem_solving': "Standard evaluation",
                'practical_application': "Standard evaluation"
            }
        }
    
    async def _call_ollama(self, prompt: str) -> str:
        """Make optimized Ollama API call."""
        logger.info(f"Making Ollama call with prompt length: {len(prompt)} chars")
        
        payload = {
            "model": ollama_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,  # Lower temperature for faster, more focused responses
                "num_predict": 250,  # Reduced tokens for faster generation
                "top_k": 20,         # More focused token selection
                "top_p": 0.8         # Faster sampling
            }
        }
        
        try:
            response = await self.ollama_client.post("/api/generate", json=payload)
            logger.info(f"Ollama response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            logger.info(f"Ollama response length: {len(response_text)} chars")
            logger.info(f"Ollama response preview: {response_text[:200]}...")
            return response_text
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    async def close(self):
        """Clean up resources."""
        await self.ollama_client.aclose()


# Global instance
comprehensive_evaluator = None


async def get_comprehensive_evaluator() -> ComprehensiveEvaluator:
    """Get or create comprehensive evaluator instance."""
    global comprehensive_evaluator
    
    if comprehensive_evaluator is None:
        comprehensive_evaluator = ComprehensiveEvaluator()
    
    return comprehensive_evaluator