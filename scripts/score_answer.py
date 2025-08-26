"""
Answer scoring module for AI Interview Coach
Provides objective, consistent scoring of interview responses using MCP context.

TUTORIAL: This script is the "AI Judge" - it provides objective numerical scores
for interview answers. Think of it as an expert evaluator who:

1. Applies consistent scoring criteria across all candidates
2. Uses weighted scoring across multiple dimensions
3. Adjusts expectations based on role and experience level
4. Provides transparent scoring rationale
5. Maintains calibration and fairness

Key Concepts:
- Multi-dimensional scoring (technical accuracy, clarity, completeness, etc.)
- Role and level-based adjustments (junior vs senior expectations)
- Weighted scoring (technical accuracy counts more than communication)
- Score normalization and calibration
- Transparent scoring rationale
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import statistics

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import ollama_config
import httpx

logger = logging.getLogger(__name__)

@dataclass
class ScoringCriteria:
    """
    TUTORIAL: Defines the scoring framework for evaluation.
    
    This is like a "digital rubric" that defines exactly how answers
    should be scored. It ensures consistency and fairness across
    all evaluations.
    """
    dimension: str              # Name of scoring dimension
    weight: float              # How much this dimension contributes (0.0-1.0)
    description: str           # What this dimension measures
    level_expectations: Dict   # Different expectations by experience level
    
@dataclass 
class DimensionScore:
    """
    TUTORIAL: Score for a single evaluation dimension.
    
    Each dimension (like "technical accuracy") gets its own score
    with detailed rationale. This makes scoring transparent and
    helps candidates understand exactly how they performed.
    """
    dimension: str              # Which dimension (e.g., "technical_accuracy")
    raw_score: float           # Base score (1.0-5.0)
    adjusted_score: float      # After role/level adjustments
    weight: float              # Contribution to overall score
    rationale: str             # Why this score was given
    evidence: List[str]        # Specific examples from the answer

@dataclass
class ScoreResult:
    """
    TUTORIAL: Complete scoring result for an answer.
    
    This packages all scoring information into a comprehensive
    result that includes both the final score and the detailed
    breakdown of how it was calculated.
    """
    question_id: str
    overall_score: float              # Final weighted score (1.0-5.0)
    dimension_scores: List[DimensionScore]  # Individual dimension scores
    role: str
    level: str
    topic: str
    difficulty: str
    
    # Scoring metadata
    adjustments_applied: Dict         # What adjustments were made
    performance_category: str         # excellent, strong, satisfactory, etc.
    scoring_rationale: str           # Overall explanation
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'question_id': self.question_id,
            'overall_score': self.overall_score,
            'dimension_scores': [asdict(score) for score in self.dimension_scores],
            'role': self.role,
            'level': self.level,
            'topic': self.topic,
            'difficulty': self.difficulty,
            'adjustments_applied': self.adjustments_applied,
            'performance_category': self.performance_category,
            'scoring_rationale': self.scoring_rationale,
            'generated_at': self.generated_at.isoformat()
        }

class AnswerScorer:
    """
    TUTORIAL: The main scoring engine for interview answers.
    
    This class implements objective, consistent scoring using the MCP
    scoring context. It's like having a standardized test grader that:
    
    1. Applies consistent criteria across all candidates
    2. Adjusts expectations based on role and level
    3. Provides detailed scoring breakdowns
    4. Maintains calibration and fairness
    
    Architecture Pattern: This follows the "Multi-Dimensional Evaluator" pattern:
    - Each dimension is scored independently
    - Scores are weighted and combined
    - Role/level adjustments are applied
    - Results are normalized and categorized
    """
    
    def __init__(self, contexts_dir: str = "contexts"):
        """
        TUTORIAL: Initialize the answer scorer.
        
        Loads the MCP scorer context which defines:
        - Scoring dimensions and weights
        - Role-based adjustments
        - Performance categories
        - Calibration guidelines
        """
        self.contexts_dir = Path(contexts_dir)
        self.scorer_context = self._load_context("scorer.context.yaml")
        
        # Extract scoring framework from context
        self.scoring_dimensions = self._load_scoring_dimensions()
        self.role_adjustments = self._load_role_adjustments()
        self.difficulty_modifiers = self._load_difficulty_modifiers()
        self.performance_categories = self._load_performance_categories()
        
        # Initialize Ollama client for AI-powered scoring
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
            logger.info(f"Loaded scorer context: {context_file}")
            return context
        except Exception as e:
            logger.error(f"Failed to load context {context_file}: {e}")
            return {}
    
    def _load_scoring_dimensions(self) -> List[ScoringCriteria]:
        """
        TUTORIAL: Load scoring dimensions from MCP context.
        
        This extracts the scoring framework (dimensions, weights, descriptions)
        from the YAML context and converts it to structured objects.
        """
        dimensions_config = self.scorer_context.get('scoring_dimensions', {})
        criteria_list = []
        
        for dim_name, dim_config in dimensions_config.items():
            criteria = ScoringCriteria(
                dimension=dim_name,
                weight=dim_config.get('weight', 20) / 100,  # Convert percentage to decimal
                description=dim_config.get('description', ''),
                level_expectations=dim_config.get('scale', {})
            )
            criteria_list.append(criteria)
        
        # Sort by weight (most important first)
        criteria_list.sort(key=lambda c: c.weight, reverse=True)
        
        logger.info(f"Loaded {len(criteria_list)} scoring dimensions")
        return criteria_list
    
    def _load_role_adjustments(self) -> Dict:
        """Load role-based scoring adjustments from context."""
        return self.scorer_context.get('role_based_adjustments', {})
    
    def _load_difficulty_modifiers(self) -> Dict:
        """Load question difficulty modifiers from context."""
        return self.scorer_context.get('question_difficulty_modifiers', {})
    
    def _load_performance_categories(self) -> Dict:
        """Load performance category definitions from context."""
        return self.scorer_context.get('performance_categories', {})
    
    async def score_answer(self,
                          question_content: str,
                          candidate_answer: str,
                          question_metadata: Dict,
                          session_context: Optional[Dict] = None) -> ScoreResult:
        """
        TUTORIAL: Main scoring method.
        
        This is the core scoring pipeline that transforms a raw answer
        into an objective, multi-dimensional score. The process:
        
        1. ANALYZE: Use AI to evaluate each scoring dimension
        2. ADJUST: Apply role/level/difficulty adjustments
        3. WEIGHT: Combine dimension scores using weights
        4. CATEGORIZE: Determine performance category
        5. RATIONALIZE: Generate explanation of the score
        
        Args:
            question_content: The original question
            candidate_answer: The candidate's response
            question_metadata: Question info (role, level, topic, difficulty)
            session_context: Optional session context
            
        Returns:
            ScoreResult with detailed scoring breakdown
        """
        
        logger.info(f"Scoring answer for {question_metadata.get('role')} {question_metadata.get('level')}")
        
        # Step 1: Score each dimension independently
        dimension_scores = await self._score_dimensions(
            question_content, candidate_answer, question_metadata, session_context
        )
        
        # Step 2: Apply role and difficulty adjustments
        adjusted_scores = self._apply_adjustments(dimension_scores, question_metadata)
        
        # Step 3: Calculate weighted overall score
        overall_score = self._calculate_overall_score(adjusted_scores)
        
        # Step 4: Determine performance category
        performance_category = self._categorize_performance(overall_score)
        
        # Step 5: Generate scoring rationale
        scoring_rationale = await self._generate_scoring_rationale(
            overall_score, adjusted_scores, question_metadata
        )
        
        # Step 6: Package results
        result = ScoreResult(
            question_id=question_metadata.get('question_id', 'unknown'),
            overall_score=overall_score,
            dimension_scores=adjusted_scores,
            role=question_metadata.get('role', 'general'),
            level=question_metadata.get('level', 'mid'),
            topic=question_metadata.get('topic', 'technical'),
            difficulty=question_metadata.get('difficulty', 'medium'),
            adjustments_applied=self._get_applied_adjustments(question_metadata),
            performance_category=performance_category,
            scoring_rationale=scoring_rationale,
            generated_at=datetime.now()
        )
        
        logger.info(f"Generated score: {overall_score:.1f}/5.0 ({performance_category})")
        return result
    
    async def _score_dimensions(self,
                               question_content: str,
                               candidate_answer: str,
                               metadata: Dict,
                               session_context: Optional[Dict] = None) -> List[DimensionScore]:
        """
        TUTORIAL: Score each dimension using AI analysis.
        
        This uses AI to evaluate the answer across each scoring dimension.
        For each dimension, we ask the AI to:
        1. Rate the answer on a 1-5 scale
        2. Provide specific rationale
        3. Cite evidence from the answer
        
        This ensures each dimension is evaluated independently and thoroughly.
        """
        
        dimension_scores = []
        
        for criteria in self.scoring_dimensions:
            try:
                # Score this specific dimension
                score_data = await self._score_single_dimension(
                    criteria, question_content, candidate_answer, metadata
                )
                
                dimension_score = DimensionScore(
                    dimension=criteria.dimension,
                    raw_score=score_data['score'],
                    adjusted_score=score_data['score'],  # Will be adjusted later
                    weight=criteria.weight,
                    rationale=score_data['rationale'],
                    evidence=score_data['evidence']
                )
                
                dimension_scores.append(dimension_score)
                
            except Exception as e:
                logger.error(f"Failed to score dimension {criteria.dimension}: {e}")
                # Fallback to neutral score
                dimension_scores.append(self._create_fallback_dimension_score(criteria))
        
        return dimension_scores
    
    async def _score_single_dimension(self,
                                    criteria: ScoringCriteria,
                                    question_content: str,
                                    candidate_answer: str,
                                    metadata: Dict) -> Dict:
        """
        TUTORIAL: Score a single dimension using AI.
        
        This creates a focused prompt that asks the AI to evaluate
        just one dimension (like "technical accuracy") and provide
        a score with detailed rationale.
        """
        
        # Build dimension-specific scoring prompt
        prompt = self._build_dimension_scoring_prompt(
            criteria, question_content, candidate_answer, metadata
        )
        
        try:
            # Get AI evaluation
            response = await self._call_ollama(prompt)
            
            # Parse response
            score_data = self._parse_dimension_score(response, criteria)
            return score_data
            
        except Exception as e:
            logger.error(f"Failed to get AI score for {criteria.dimension}: {e}")
            # Return default score
            return {
                'score': 3.0,
                'rationale': f"Unable to evaluate {criteria.dimension}",
                'evidence': []
            }
    
    def _build_dimension_scoring_prompt(self,
                                      criteria: ScoringCriteria,
                                      question_content: str,
                                      candidate_answer: str,
                                      metadata: Dict) -> str:
        """
        TUTORIAL: Build dimension-specific scoring prompt.
        
        This creates a focused prompt that tells the AI exactly how
        to evaluate one specific dimension. The prompt includes:
        1. Context about the interview and question
        2. Definition of the scoring dimension
        3. Scoring scale and criteria
        4. Request for structured output
        """
        
        role = metadata.get('role', 'general')
        level = metadata.get('level', 'mid')
        topic = metadata.get('topic', 'technical')
        
        # Get level-specific expectations for this dimension
        level_expectations = criteria.level_expectations.get(str(5), "Excellent performance")
        
        prompt = f"""
You are an expert technical evaluator scoring interview answers.

CONTEXT:
- Role: {role}
- Level: {level}  
- Topic: {topic}
- Question: {question_content}
- Answer: {candidate_answer}

SCORING DIMENSION: {criteria.dimension}
Description: {criteria.description}
Weight: {criteria.weight * 100}% of total score

EVALUATION TASK:
Score this answer ONLY on the "{criteria.dimension}" dimension using a 1-5 scale:
- 5: {criteria.level_expectations.get('5', 'Excellent')}
- 4: {criteria.level_expectations.get('4', 'Good')}
- 3: {criteria.level_expectations.get('3', 'Satisfactory')}
- 2: {criteria.level_expectations.get('2', 'Needs Improvement')}
- 1: {criteria.level_expectations.get('1', 'Unsatisfactory')}

Consider {level}-level expectations for {role} role.

Provide your response in JSON format:
{{
  "score": <number 1-5>,
  "rationale": "<specific explanation for this score>",
  "evidence": ["<specific examples from the answer>"]
}}

Focus ONLY on {criteria.dimension}. Be specific and cite evidence.
"""
        return prompt
    
    def _parse_dimension_score(self, response: str, criteria: ScoringCriteria) -> Dict:
        """
        TUTORIAL: Parse AI response for dimension scoring.
        
        Extracts the score, rationale, and evidence from the AI's JSON response.
        Includes error handling for malformed responses.
        """
        try:
            score_data = json.loads(response.strip())
            
            # Validate score is in range
            score = float(score_data.get('score', 3.0))
            score = max(1.0, min(5.0, score))  # Clamp to valid range
            
            return {
                'score': score,
                'rationale': score_data.get('rationale', f'Score for {criteria.dimension}'),
                'evidence': score_data.get('evidence', [])
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse dimension score: {e}")
            # Extract numeric score with regex as fallback
            import re
            score_match = re.search(r'["\']score["\']\s*:\s*(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                score = max(1.0, min(5.0, score))
                return {
                    'score': score,
                    'rationale': f'Evaluated {criteria.dimension}',
                    'evidence': []
                }
            
            # Final fallback
            return {
                'score': 3.0,
                'rationale': f'Unable to parse {criteria.dimension} evaluation',
                'evidence': []
            }
    
    def _create_fallback_dimension_score(self, criteria: ScoringCriteria) -> DimensionScore:
        """Create fallback dimension score when AI evaluation fails."""
        return DimensionScore(
            dimension=criteria.dimension,
            raw_score=3.0,
            adjusted_score=3.0,
            weight=criteria.weight,
            rationale=f"Unable to evaluate {criteria.dimension}",
            evidence=[]
        )
    
    def _apply_adjustments(self,
                          dimension_scores: List[DimensionScore],
                          metadata: Dict) -> List[DimensionScore]:
        """
        TUTORIAL: Apply role, level, and difficulty adjustments.
        
        This modifies the raw scores based on:
        1. Role expectations (different roles have different priorities)
        2. Level expectations (junior vs senior standards)
        3. Question difficulty (harder questions get more lenient scoring)
        
        The goal is to ensure fair comparison across different contexts.
        """
        
        role = metadata.get('role', 'general')
        level = metadata.get('level', 'mid')
        difficulty = metadata.get('difficulty', 'medium')
        
        adjusted_scores = []
        
        for score in dimension_scores:
            # Start with raw score
            adjusted_score = score.raw_score
            
            # Apply level adjustment
            level_modifier = self._get_level_modifier(level)
            adjusted_score *= level_modifier
            
            # Apply difficulty adjustment  
            difficulty_modifier = self._get_difficulty_modifier(difficulty)
            adjusted_score *= difficulty_modifier
            
            # Apply role-specific adjustments
            role_modifier = self._get_role_modifier(role, score.dimension)
            adjusted_score *= role_modifier
            
            # Clamp to valid range
            adjusted_score = max(1.0, min(5.0, adjusted_score))
            
            # Create adjusted score object
            adjusted_score_obj = DimensionScore(
                dimension=score.dimension,
                raw_score=score.raw_score,
                adjusted_score=adjusted_score,
                weight=score.weight,
                rationale=score.rationale,
                evidence=score.evidence
            )
            
            adjusted_scores.append(adjusted_score_obj)
        
        return adjusted_scores
    
    def _get_level_modifier(self, level: str) -> float:
        """
        TUTORIAL: Get scoring modifier based on experience level.
        
        Different levels have different expectations:
        - Junior: More lenient scoring (encourage growth)
        - Senior: Standard scoring
        - Principal: Higher expectations
        """
        level_modifiers = {
            'junior': 1.2,      # 20% more lenient
            'mid': 1.0,         # Standard expectations
            'senior': 0.95,     # 5% higher expectations
            'principal': 0.9    # 10% higher expectations
        }
        return level_modifiers.get(level, 1.0)
    
    def _get_difficulty_modifier(self, difficulty: str) -> float:
        """Get scoring modifier based on question difficulty."""
        difficulty_config = self.difficulty_modifiers
        return difficulty_config.get(difficulty, {}).get('modifier', 1.0)
    
    def _get_role_modifier(self, role: str, dimension: str) -> float:
        """
        TUTORIAL: Get role-specific dimension modifiers.
        
        Different roles emphasize different dimensions:
        - Data Scientists: Statistics and analysis skills matter more
        - DevOps: System thinking and reliability focus
        - Frontend: Communication and user experience focus
        """
        role_emphasis = {
            'data_scientist': {
                'technical_accuracy': 1.1,     # Stats accuracy is crucial
                'problem_solving': 1.1,        # Analytical thinking
                'practical_application': 1.0
            },
            'software_engineer': {
                'technical_accuracy': 1.1,     # Code correctness matters
                'problem_solving': 1.1,        # Algorithm design
                'clarity_communication': 1.0
            },
            'devops': {
                'technical_accuracy': 1.1,     # System reliability
                'problem_solving': 1.1,        # Troubleshooting
                'practical_application': 1.1   # Real-world experience
            }
        }
        
        role_modifiers = role_emphasis.get(role, {})
        return role_modifiers.get(dimension, 1.0)
    
    def _calculate_overall_score(self, dimension_scores: List[DimensionScore]) -> float:
        """
        TUTORIAL: Calculate weighted overall score.
        
        This combines all dimension scores using their weights to produce
        a single overall score. The formula is:
        
        Overall = Σ(dimension_score * weight) / Σ(weights)
        
        This ensures that more important dimensions (higher weights)
        contribute more to the final score.
        """
        
        if not dimension_scores:
            return 3.0  # Default neutral score
        
        # Calculate weighted sum
        weighted_sum = sum(score.adjusted_score * score.weight for score in dimension_scores)
        total_weight = sum(score.weight for score in dimension_scores)
        
        if total_weight == 0:
            return 3.0
        
        # Calculate weighted average
        overall_score = weighted_sum / total_weight
        
        # Round to 1 decimal place
        return round(overall_score, 1)
    
    def _categorize_performance(self, overall_score: float) -> str:
        """
        TUTORIAL: Categorize performance based on overall score.
        
        This converts the numerical score into a human-readable category
        that makes it easier to understand performance levels.
        """
        
        for category, config in self.performance_categories.items():
            score_range = config.get('range', '0.0 - 5.0')
            min_score, max_score = map(float, score_range.split(' - '))
            
            if min_score <= overall_score <= max_score:
                return category
        
        # Fallback categorization
        if overall_score >= 4.5:
            return "excellent"
        elif overall_score >= 3.5:
            return "strong"
        elif overall_score >= 2.5:
            return "satisfactory"
        elif overall_score >= 1.5:
            return "needs_improvement"
        else:
            return "unsatisfactory"
    
    async def _generate_scoring_rationale(self,
                                        overall_score: float,
                                        dimension_scores: List[DimensionScore],
                                        metadata: Dict) -> str:
        """
        TUTORIAL: Generate explanation for the overall score.
        
        This creates a human-readable explanation of why the score
        was assigned, referencing the dimension scores and any
        adjustments that were applied.
        """
        
        # Build rationale prompt
        prompt = f"""
You are explaining an interview scoring result to provide transparency.

OVERALL SCORE: {overall_score}/5.0
ROLE: {metadata.get('role')}
LEVEL: {metadata.get('level')}

DIMENSION SCORES:
{self._format_dimension_scores_for_rationale(dimension_scores)}

Generate a brief, professional explanation (2-3 sentences) that:
1. States the overall score and performance level
2. Highlights the strongest dimension(s)
3. Notes any significant strengths or areas that influenced the score

Be objective and factual. Focus on the scoring rationale, not feedback.
"""
        
        try:
            rationale = await self._call_ollama(prompt)
            return rationale
        except Exception as e:
            logger.error(f"Failed to generate scoring rationale: {e}")
            return self._generate_template_rationale(overall_score, dimension_scores)
    
    def _format_dimension_scores_for_rationale(self, dimension_scores: List[DimensionScore]) -> str:
        """Format dimension scores for inclusion in rationale prompt."""
        formatted_scores = []
        for score in dimension_scores:
            formatted_scores.append(
                f"- {score.dimension}: {score.adjusted_score:.1f}/5.0 ({score.rationale})"
            )
        return '\n'.join(formatted_scores)
    
    def _generate_template_rationale(self,
                                   overall_score: float,
                                   dimension_scores: List[DimensionScore]) -> str:
        """Generate template-based rationale as fallback."""
        category = self._categorize_performance(overall_score)
        
        # Find strongest dimension
        strongest_dim = max(dimension_scores, key=lambda s: s.adjusted_score)
        
        return (f"Overall score of {overall_score}/5.0 reflects {category} performance. "
                f"Strongest area was {strongest_dim.dimension} ({strongest_dim.adjusted_score:.1f}/5.0). "
                f"Score considers role expectations and question difficulty.")
    
    def _get_applied_adjustments(self, metadata: Dict) -> Dict:
        """Get summary of adjustments applied to the score."""
        return {
            'level_adjustment': self._get_level_modifier(metadata.get('level', 'mid')),
            'difficulty_adjustment': self._get_difficulty_modifier(metadata.get('difficulty', 'medium')),
            'role': metadata.get('role', 'general')
        }
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation."""
        payload = {
            "model": ollama_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,    # Lower temperature for consistent scoring
                "num_predict": 200,     # Shorter rationale for faster scoring
                "top_p": 0.8          # More focused responses
            }
        }
        
        response = await self.ollama_client.post("/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    def get_score_statistics(self, score_results: List[ScoreResult]) -> Dict:
        """
        TUTORIAL: Generate scoring statistics across multiple results.
        
        This analyzes patterns across all scored answers to provide insights
        like average scores, score distributions, and dimension performance.
        """
        if not score_results:
            return {'error': 'No score results to analyze'}
        
        # Overall statistics
        overall_scores = [result.overall_score for result in score_results]
        avg_score = statistics.mean(overall_scores)
        score_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
        
        # Dimension statistics
        dimension_stats = {}
        for result in score_results:
            for dim_score in result.dimension_scores:
                dim_name = dim_score.dimension
                if dim_name not in dimension_stats:
                    dimension_stats[dim_name] = []
                dimension_stats[dim_name].append(dim_score.adjusted_score)
        
        # Calculate dimension averages
        dimension_averages = {
            dim: statistics.mean(scores)
            for dim, scores in dimension_stats.items()
        }
        
        # Performance category distribution
        categories = [result.performance_category for result in score_results]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        return {
            'total_questions': len(score_results),
            'average_score': round(avg_score, 2),
            'score_std_dev': round(score_std, 2),
            'score_range': f"{min(overall_scores):.1f} - {max(overall_scores):.1f}",
            'dimension_averages': {k: round(v, 2) for k, v in dimension_averages.items()},
            'performance_distribution': category_counts,
            'strongest_dimension': max(dimension_averages.items(), key=lambda x: x[1])[0],
            'weakest_dimension': min(dimension_averages.items(), key=lambda x: x[1])[0]
        }
    
    async def close(self):
        """Clean up resources."""
        await self.ollama_client.aclose()

# CLI interface for testing
async def main():
    """
    TUTORIAL: CLI interface for testing answer scoring.
    
    This allows you to test the scoring system with sample questions and answers:
    python score_answer.py --question "Explain algorithms" --answer "Algorithms are procedures"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Answer Scorer for AI Interview Coach")
    parser.add_argument("--question", required=True, help="Question content")
    parser.add_argument("--answer", required=True, help="Candidate's answer")
    parser.add_argument("--role", default="software_engineer", help="Interview role")
    parser.add_argument("--level", default="mid", help="Experience level")
    parser.add_argument("--topic", default="technical", help="Question topic")
    parser.add_argument("--difficulty", default="medium", help="Question difficulty")
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scorer
    scorer = AnswerScorer()
    
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
        # Generate score
        print("Scoring answer...")
        score_result = await scorer.score_answer(
            question_content=args.question,
            candidate_answer=args.answer,
            question_metadata=metadata
        )
        
        print("\n" + "="*50)
        print("SCORING RESULT")
        print("="*50)
        print(f"Overall Score: {score_result.overall_score}/5.0")
        print(f"Performance Category: {score_result.performance_category}")
        print(f"Question: {args.question}")
        print(f"Answer: {args.answer}")
        
        print("\n" + "="*50)
        print("DIMENSION BREAKDOWN")
        print("="*50)
        for dim_score in score_result.dimension_scores:
            print(f"{dim_score.dimension}:")
            print(f"  Score: {dim_score.adjusted_score:.1f}/5.0 (weight: {dim_score.weight*100:.0f}%)")
            print(f"  Rationale: {dim_score.rationale}")
            if dim_score.evidence:
                print(f"  Evidence: {', '.join(dim_score.evidence)}")
        
        print("\n" + "="*50)
        print("SCORING RATIONALE")
        print("="*50)
        print(score_result.scoring_rationale)
        
        print("\n" + "="*50)
        print("ADJUSTMENTS APPLIED")
        print("="*50)
        for adjustment, value in score_result.adjustments_applied.items():
            print(f"{adjustment}: {value}")
        
    except Exception as e:
        print(f"Error scoring answer: {e}")
        
    finally:
        await scorer.close()

if __name__ == "__main__":
    asyncio.run(main())