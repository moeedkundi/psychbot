"""
Report generation module for AI Interview Coach
Creates comprehensive interview reports combining scoring, feedback, and analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import statistics
from jinja2 import Template, Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64

# Import our other modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.score_answer import ScoreResult, AnswerScorer
from scripts.generate_feedback import FeedbackResult, FeedbackGenerator
from scripts.ask_question import InterviewSession
from src.config import ollama_config
import httpx

logger = logging.getLogger(__name__)

@dataclass
class InterviewData:
    """Complete interview data structure for report generation."""
    session_info: Dict              # Basic session information
    questions_and_answers: List[Dict]  # Q&A pairs with metadata
    scores: List[ScoreResult]       # Scoring results for each answer
    feedback: List[FeedbackResult]  # Feedback for each answer
    performance_summary: Dict       # Aggregated performance metrics
    generated_at: datetime

@dataclass 
class ReportConfiguration:
    """Configuration for report generation options."""
    report_type: str                # "candidate", "interviewer", "summary"
    format: str                     # "markdown", "pdf", "json", "html"
    include_detailed_scores: bool
    include_improvement_suggestions: bool
    include_charts: bool
    include_raw_answers: bool
    template_name: Optional[str] = None

class ReportGenerator:
    """Main report generation engine for interview results."""
    
    def __init__(self, 
                 templates_dir: str = "templates",
                 output_dir: str = "reports"):
        """Initialize the report generator with template and output directories."""
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 template engine
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Initialize AI client for report enhancement
        self.ollama_client = httpx.AsyncClient(
            base_url=ollama_config.base_url,
            timeout=ollama_config.timeout
        )
        
        logger.info("Report generator initialized")
    
    async def generate_interview_report(self,
                                      session_data: Dict,
                                      scores: List[ScoreResult],
                                      feedback: List[FeedbackResult],
                                      config: ReportConfiguration) -> Dict:
        """Generate comprehensive interview report from session data."""
        
        logger.info(f"Generating {config.report_type} report in {config.format} format")
        
        # Step 1: Aggregate all interview data
        interview_data = await self._aggregate_interview_data(
            session_data, scores, feedback
        )
        
        # Step 2: Analyze performance patterns
        analysis = await self._analyze_performance(interview_data)
        
        # Step 3: Generate visualizations (if requested)
        charts = {}
        if config.include_charts:
            charts = await self._generate_charts(interview_data, analysis)
        
        # Step 4: Generate insights and recommendations
        insights = await self._generate_insights(interview_data, analysis, config.report_type)
        
        # Step 5: Apply template and format report
        report_content = await self._format_report(
            interview_data, analysis, charts, insights, config
        )
        
        # Step 6: Save report file
        report_file = await self._save_report(report_content, config, interview_data)
        
        result = {
            'report_file': str(report_file),
            'format': config.format,
            'type': config.report_type,
            'content': report_content,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'session_id': session_data.get('session_id'),
                'candidate': session_data.get('candidate_name'),
                'role': session_data.get('role'),
                'total_questions': len(scores),
                'overall_score': analysis.get('overall_performance', {}).get('average_score', 0)
            }
        }
        
        logger.info(f"Report generated successfully: {report_file}")
        return result
    
    async def _aggregate_interview_data(self,
                                      session_data: Dict,
                                      scores: List[ScoreResult],
                                      feedback: List[FeedbackResult]) -> InterviewData:
        """Aggregate all interview data into unified structure."""
        
        # Combine questions, answers, scores, and feedback
        qa_data = []
        
        for i, (score, fb) in enumerate(zip(scores, feedback)):
            # Handle both dict and object types
            if isinstance(score, dict):
                question_id = f"q_{i+1}"
                topic = "general"
                difficulty = "medium"
                overall_score = score.get('overall_score', 0.0)
                # Convert flat category_scores to expected format
                category_scores = score.get('category_scores', {})
                score_breakdown = {
                    dim: {'score': score_val, 'weight': 1.0, 'rationale': f'Score: {score_val}'}
                    for dim, score_val in category_scores.items()
                }
            else:
                question_id = score.question_id
                topic = score.topic
                difficulty = score.difficulty
                overall_score = score.overall_score
                score_breakdown = {
                    dim.dimension: {
                        'score': dim.adjusted_score,
                        'weight': dim.weight,
                        'rationale': dim.rationale
                    }
                    for dim in score.dimension_scores
                }
            
            if isinstance(fb, dict):
                overall_rating = fb.get('overall_rating', 0.0)
                question_content = "Question content"
                candidate_answer = "Candidate answer"
                strengths = ["Response provided"]
                improvements = ["Continue learning"]
                suggestions = ["Practice more"]
                resources = ["Study materials"]
            else:
                overall_rating = fb.overall_rating
                question_content = fb.analysis.question_content
                candidate_answer = fb.analysis.candidate_answer
                strengths = fb.analysis.strengths
                improvements = fb.analysis.improvement_areas
                suggestions = fb.analysis.specific_suggestions
                resources = fb.analysis.recommended_resources
            
            qa_item = {
                'question_number': i + 1,
                'question_id': question_id,
                'question_content': question_content,
                'candidate_answer': candidate_answer,
                'topic': topic,
                'difficulty': difficulty,
                'score': overall_score,
                'score_breakdown': score_breakdown,
                'feedback': {
                    'overall_rating': overall_rating,
                    'strengths': strengths,
                    'improvements': improvements,
                    'suggestions': suggestions,
                    'resources': resources
                }
            }
            qa_data.append(qa_item)
        
        # Calculate performance summary
        performance_summary = self._calculate_performance_summary(scores, feedback)
        
        return InterviewData(
            session_info=session_data,
            questions_and_answers=qa_data,
            scores=scores,
            feedback=feedback,
            performance_summary=performance_summary,
            generated_at=datetime.now()
        )
    
    def _calculate_performance_summary(self,
                                     scores: List[ScoreResult],
                                     feedback: List[FeedbackResult]) -> Dict:
        """Calculate aggregate performance metrics across all questions."""
        
        if not scores:
            return {}
        
        # Overall score statistics - handle both dict and object types
        overall_scores = []
        for score in scores:
            if isinstance(score, dict):
                overall_scores.append(score.get('overall_score', 0.0))
            else:
                overall_scores.append(score.overall_score)
        avg_score = statistics.mean(overall_scores)
        score_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
        
        # Dimension performance analysis
        dimension_performance = {}
        for score in scores:
            if isinstance(score, dict):
                category_scores = score.get('category_scores', {})
                for dim_name, dim_score in category_scores.items():
                    if dim_name not in dimension_performance:
                        dimension_performance[dim_name] = []
                    # Handle both dict and number formats
                    score_value = dim_score if isinstance(dim_score, (int, float)) else dim_score.get('score', 0.0) if isinstance(dim_score, dict) else 0.0
                    dimension_performance[dim_name].append(score_value)
            else:
                for dim_score in score.dimension_scores:
                    dim_name = dim_score.dimension
                    if dim_name not in dimension_performance:
                        dimension_performance[dim_name] = []
                    dimension_performance[dim_name].append(dim_score.adjusted_score)
        
        dimension_averages = {
            dim: statistics.mean(scores_list)
            for dim, scores_list in dimension_performance.items()
        }
        
        # Topic performance analysis
        topic_performance = {}
        for score in scores:
            if isinstance(score, dict):
                topic = "general"  # Default topic for dict format
                overall_score = score.get('overall_score', 0.0)
            else:
                topic = score.topic
                overall_score = score.overall_score
            
            if topic not in topic_performance:
                topic_performance[topic] = []
            topic_performance[topic].append(overall_score)
        
        topic_averages = {
            topic: statistics.mean(scores_list)
            for topic, scores_list in topic_performance.items()
        }
        
        # Performance category distribution
        categories = []
        for score in scores:
            if isinstance(score, dict):
                overall_score = score.get('overall_score', 0.0)
                category = 'good' if overall_score >= 3.0 else 'needs_improvement'
            else:
                category = score.performance_category
            categories.append(category)
        category_distribution = {cat: categories.count(cat) for cat in set(categories)}
        
        # Improvement themes analysis
        all_improvements = []
        all_suggestions = []
        for fb in feedback:
            if isinstance(fb, dict):
                all_improvements.extend(['Continue learning'])
                all_suggestions.extend(['Practice more'])
            else:
                all_improvements.extend(fb.analysis.improvement_areas)
                all_suggestions.extend(fb.analysis.specific_suggestions)
        
        return {
            'overall_performance': {
                'average_score': round(avg_score, 2),
                'score_range': f"{min(overall_scores):.1f} - {max(overall_scores):.1f}",
                'score_std_dev': round(score_std, 2),
                'total_questions': len(scores)
            },
            'dimension_performance': {k: round(v, 2) for k, v in dimension_averages.items()},
            'topic_performance': {k: round(v, 2) for k, v in topic_averages.items()},
            'strongest_dimension': max(dimension_averages.items(), key=lambda x: x[1])[0] if dimension_averages else None,
            'weakest_dimension': min(dimension_averages.items(), key=lambda x: x[1])[0] if dimension_averages else None,
            'strongest_topic': max(topic_averages.items(), key=lambda x: x[1])[0] if topic_averages else None,
            'focus_topic': min(topic_averages.items(), key=lambda x: x[1])[0] if topic_averages else None,
            'performance_distribution': category_distribution,
            'improvement_themes': self._extract_common_themes(all_improvements),
            'suggestion_themes': self._extract_common_themes(all_suggestions)
        }
    
    def _extract_common_themes(self, text_list: List[str], min_frequency: int = 2) -> List[str]:
        """Extract common themes from a list of text items."""
        if not text_list:
            return []
        
        # Simple word frequency analysis
        word_counts = {}
        for text in text_list:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return words that appear frequently
        common_themes = [
            word for word, count in word_counts.items() 
            if count >= min_frequency
        ]
        
        return sorted(common_themes, key=lambda w: word_counts[w], reverse=True)[:5]
    
    async def _analyze_performance(self, interview_data: InterviewData) -> Dict:
        """Analyze performance patterns using AI and statistical methods."""
        
        analysis_prompt = self._build_performance_analysis_prompt(interview_data)
        
        try:
            ai_analysis = await self._call_ollama(analysis_prompt)
            parsed_analysis = self._parse_ai_analysis(ai_analysis)
            
            # Combine with statistical analysis
            combined_analysis = {
                **interview_data.performance_summary,
                'ai_insights': parsed_analysis,
                'performance_trajectory': self._analyze_performance_trajectory(interview_data),
                'knowledge_gaps': self._identify_knowledge_gaps(interview_data),
                'recommendations': self._generate_recommendations(interview_data)
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Failed to generate AI analysis: {e}")
            # Return statistical analysis only
            return interview_data.performance_summary
    
    def _build_performance_analysis_prompt(self, interview_data: InterviewData) -> str:
        """Build prompt for AI-powered performance analysis."""
        session_info = interview_data.session_info
        qa_summary = self._summarize_qa_for_analysis(interview_data.questions_and_answers)
        performance_summary = interview_data.performance_summary
        
        prompt = f"""
You are an expert technical interview analyst reviewing a candidate's performance.

CANDIDATE PROFILE:
- Role: {session_info.get('role')}
- Level: {session_info.get('level')}
- Total Questions: {len(interview_data.questions_and_answers)}

PERFORMANCE SUMMARY:
- Average Score: {performance_summary['overall_performance']['average_score']}/5.0
- Strongest Area: {performance_summary.get('strongest_dimension')}
- Focus Area: {performance_summary.get('weakest_dimension')}
- Topic Performance: {performance_summary['topic_performance']}

QUESTION BREAKDOWN:
{qa_summary}

Analyze this interview performance and provide insights in JSON format:
{{
  "overall_assessment": "<2-3 sentence overall evaluation>",
  "key_strengths": ["<list of main strengths>"],
  "primary_concerns": ["<list of main concerns>"],
  "learning_trajectory": "<improving/stable/declining>",
  "readiness_level": "<ready/needs_development/not_ready>",
  "strategic_recommendations": ["<high-level recommendations>"]
}}

Focus on patterns across questions, not individual responses. Be objective and specific.
"""
        return prompt
    
    def _summarize_qa_for_analysis(self, qa_data: List[Dict]) -> str:
        """Summarize Q&A data for AI analysis prompt."""
        summary_lines = []
        for qa in qa_data:
            summary_lines.append(
                f"Q{qa['question_number']} ({qa['topic']}, {qa['difficulty']}): "
                f"Score {qa['score']:.1f}/5.0 - {', '.join(qa['feedback']['strengths'][:2])}"
            )
        return '\n'.join(summary_lines)
    
    def _parse_ai_analysis(self, analysis_text: str) -> Dict:
        """Parse AI analysis response into structured data."""
        try:
            return json.loads(analysis_text.strip())
        except json.JSONDecodeError:
            logger.error("Failed to parse AI analysis as JSON")
            return {
                'overall_assessment': 'Unable to generate detailed analysis',
                'key_strengths': [],
                'primary_concerns': [],
                'learning_trajectory': 'stable',
                'readiness_level': 'needs_development',
                'strategic_recommendations': []
            }
    
    def _analyze_performance_trajectory(self, interview_data: InterviewData) -> str:
        """Analyze if performance improved, declined, or stayed stable during interview."""
        scores = [qa['score'] for qa in interview_data.questions_and_answers]
        
        if len(scores) < 3:
            return 'insufficient_data'
        
        # Simple linear trend analysis
        first_half_avg = statistics.mean(scores[:len(scores)//2])
        second_half_avg = statistics.mean(scores[len(scores)//2:])
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.3:
            return 'improving'
        elif diff < -0.3:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_knowledge_gaps(self, interview_data: InterviewData) -> List[str]:
        """Identify key knowledge gaps based on low scores and feedback."""
        gaps = []
        
        # Find topics with low scores
        topic_scores = {}
        for qa in interview_data.questions_and_answers:
            topic = qa['topic']
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(qa['score'])
        
        for topic, scores in topic_scores.items():
            avg_score = statistics.mean(scores)
            if avg_score < 3.0:
                gaps.append(f"{topic} (avg: {avg_score:.1f}/5.0)")
        
        return gaps
    
    def _generate_recommendations(self, interview_data: InterviewData) -> List[str]:
        """Generate strategic recommendations based on performance analysis."""
        recommendations = []
        
        performance = interview_data.performance_summary
        avg_score = performance['overall_performance']['average_score']
        
        # Overall performance recommendations
        if avg_score >= 4.0:
            recommendations.append("Strong overall performance - ready for role responsibilities")
        elif avg_score >= 3.0:
            recommendations.append("Solid foundation with targeted development needed")
        else:
            recommendations.append("Significant skill development required before role readiness")
        
        # Dimension-specific recommendations
        weakest_dim = performance.get('weakest_dimension')
        if weakest_dim:
            recommendations.append(f"Focus development on {weakest_dim} skills")
        
        # Topic-specific recommendations
        focus_topic = performance.get('focus_topic')
        if focus_topic:
            recommendations.append(f"Prioritize learning in {focus_topic} domain")
        
        return recommendations
    
    async def _generate_charts(self, interview_data: InterviewData, analysis: Dict) -> Dict:
        """Generate visualizations for the report."""
        
        charts = {}
        
        try:
            # Set style for consistent appearance
            plt.style.use('seaborn-v0_8')
            
            # Chart 1: Score distribution
            charts['score_distribution'] = self._create_score_distribution_chart(interview_data)
            
            # Chart 2: Dimension performance radar
            charts['dimension_radar'] = self._create_dimension_radar_chart(interview_data)
            
            # Chart 3: Topic performance comparison
            charts['topic_performance'] = self._create_topic_performance_chart(interview_data)
            
            # Chart 4: Performance trajectory
            if len(interview_data.questions_and_answers) > 2:
                charts['performance_trajectory'] = self._create_trajectory_chart(interview_data)
            
            logger.info(f"Generated {len(charts)} charts for report")
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
            charts['error'] = str(e)
        
        return charts
    
    def _create_score_distribution_chart(self, interview_data: InterviewData) -> str:
        """Create score distribution chart and return as base64 string."""
        scores = [qa['score'] for qa in interview_data.questions_and_answers]
        questions = [f"Q{qa['question_number']}" for qa in interview_data.questions_and_answers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(questions, scores, color='steelblue', alpha=0.7)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{score:.1f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 5)
        ax.set_ylabel('Score')
        ax.set_title('Score Distribution Across Questions')
        ax.grid(axis='y', alpha=0.3)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def _create_dimension_radar_chart(self, interview_data: InterviewData) -> str:
        """Create radar chart for dimension performance."""
        # Aggregate dimension scores
        dim_scores = {}
        for qa in interview_data.questions_and_answers:
            for dim, data in qa['score_breakdown'].items():
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(data['score'])
        
        # Calculate averages
        dim_averages = {dim: statistics.mean(scores) for dim, scores in dim_scores.items()}
        
        # Create radar chart
        dimensions = list(dim_averages.keys())
        values = list(dim_averages.values())
        
        # Close the plot
        dimensions += [dimensions[0]]
        values += [values[0]]
        
        angles = [n / len(dim_averages) * 2 * 3.14159 for n in range(len(dim_averages))]
        angles += [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions[:-1])
        ax.set_ylim(0, 5)
        ax.set_title('Performance by Dimension', pad=20)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def _create_topic_performance_chart(self, interview_data: InterviewData) -> str:
        """Create topic performance comparison chart."""
        topic_scores = {}
        for qa in interview_data.questions_and_answers:
            topic = qa['topic']
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(qa['score'])
        
        topics = list(topic_scores.keys())
        averages = [statistics.mean(scores) for scores in topic_scores.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(topics, averages, color='lightcoral', alpha=0.7)
        
        # Add score labels
        for bar, avg in zip(bars, averages):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{avg:.1f}', ha='left', va='center')
        
        ax.set_xlim(0, 5)
        ax.set_xlabel('Average Score')
        ax.set_title('Performance by Topic Area')
        ax.grid(axis='x', alpha=0.3)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    def _create_trajectory_chart(self, interview_data: InterviewData) -> str:
        """Create performance trajectory chart."""
        scores = [qa['score'] for qa in interview_data.questions_and_answers]
        questions = list(range(1, len(scores) + 1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(questions, scores, 'o-', linewidth=2, markersize=8, color='darkgreen')
        
        # Add trend line
        import numpy as np
        z = np.polyfit(questions, scores, 1)
        p = np.poly1d(z)
        ax.plot(questions, p(questions), "--", alpha=0.7, color='red', 
                label=f'Trend (slope: {z[0]:.2f})')
        
        ax.set_ylim(0, 5)
        ax.set_xlabel('Question Number')
        ax.set_ylabel('Score')
        ax.set_title('Performance Trajectory')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
    
    async def _generate_insights(self,
                                interview_data: InterviewData,
                                analysis: Dict,
                                report_type: str) -> Dict:
        """Generate role-specific insights and recommendations."""
        
        insights_prompt = self._build_insights_prompt(interview_data, analysis, report_type)
        
        try:
            ai_insights = await self._call_ollama(insights_prompt)
            parsed_insights = self._parse_insights_response(ai_insights)
            
            return parsed_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return self._generate_fallback_insights(interview_data, analysis, report_type)
    
    def _build_insights_prompt(self,
                              interview_data: InterviewData,
                              analysis: Dict,
                              report_type: str) -> str:
        """Build prompt for generating insights and recommendations."""
        
        session_info = interview_data.session_info
        performance = interview_data.performance_summary
        
        audience_context = {
            'candidate': 'for the candidate to understand their performance and improve',
            'interviewer': 'for the interviewer to make hiring decisions',
            'summary': 'for management to understand candidate assessment results'
        }
        
        prompt = f"""
Generate interview insights and recommendations {audience_context.get(report_type, '')}.

INTERVIEW CONTEXT:
- Role: {session_info.get('role')}
- Level: {session_info.get('level')}
- Questions: {len(interview_data.questions_and_answers)}
- Overall Score: {performance['overall_performance']['average_score']}/5.0

PERFORMANCE ANALYSIS:
- Strongest Area: {performance.get('strongest_dimension')}
- Focus Area: {performance.get('weakest_dimension')}
- Performance Trajectory: {analysis.get('performance_trajectory', 'stable')}

Provide insights in JSON format:
{{
  "key_insights": ["<list of main insights>"],
  "recommendations": ["<specific actionable recommendations>"],
  "next_steps": ["<concrete next steps>"],
  "timeline": "<suggested timeline for improvement if applicable>"
}}

Tailor the tone and focus for a {report_type} audience.
"""
        return prompt
    
    def _parse_insights_response(self, response: str) -> Dict:
        """Parse AI insights response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                'key_insights': ['Performance analysis completed'],
                'recommendations': ['Continue developing technical skills'],
                'next_steps': ['Review feedback and focus on improvement areas'],
                'timeline': '3-6 months for significant improvement'
            }
    
    def _generate_fallback_insights(self,
                                   interview_data: InterviewData,
                                   analysis: Dict,
                                   report_type: str) -> Dict:
        """Generate basic insights when AI generation fails."""
        performance = interview_data.performance_summary
        avg_score = performance['overall_performance']['average_score']
        
        insights = {
            'key_insights': [
                f"Overall performance score: {avg_score:.1f}/5.0",
                f"Strongest area: {performance.get('strongest_dimension', 'N/A')}",
                f"Focus area: {performance.get('weakest_dimension', 'N/A')}"
            ],
            'recommendations': [
                "Review detailed feedback for each question",
                "Focus development on lowest-scoring dimensions",
                "Practice explaining technical concepts clearly"
            ],
            'next_steps': [
                "Study improvement areas identified in feedback",
                "Practice with similar technical questions",
                "Seek additional learning resources"
            ],
            'timeline': "2-4 weeks for focused improvement"
        }
        
        return insights
    
    async def _format_report(self,
                           interview_data: InterviewData,
                           analysis: Dict,
                           charts: Dict,
                           insights: Dict,
                           config: ReportConfiguration) -> str:
        """Format the report using templates."""
        
        # Prepare template data
        template_data = {
            'session_info': interview_data.session_info,
            'performance_summary': interview_data.performance_summary,
            'questions_and_answers': interview_data.questions_and_answers,
            'analysis': analysis,
            'insights': insights,
            'charts': charts,
            'config': asdict(config),
            'generated_at': datetime.now(),
            'include_charts': config.include_charts,
            'include_detailed_scores': config.include_detailed_scores,
            'include_raw_answers': config.include_raw_answers
        }
        
        # Determine template name
        template_name = config.template_name or f"{config.report_type}_{config.format}.j2"
        
        try:
            # Try to load custom template
            template = self.template_env.get_template(template_name)
            report_content = template.render(**template_data)
            
        except Exception as e:
            logger.warning(f"Failed to load template {template_name}: {e}")
            # Use built-in template
            report_content = self._generate_builtin_template(template_data, config)
        
        return report_content
    
    def _generate_builtin_template(self, data: Dict, config: ReportConfiguration) -> str:
        """Generate report using built-in template when custom template fails."""
        
        session = data['session_info']
        performance = data['performance_summary']
        insights = data['insights']
        
        # Basic markdown template
        report = f"""
# Interview Report

**Candidate:** {session.get('candidate_name', 'Anonymous')}  
**Role:** {session.get('role')}  
**Level:** {session.get('level')}  
**Date:** {data['generated_at'].strftime('%Y-%m-%d %H:%M')}

## Performance Summary

- **Overall Score:** {performance['overall_performance']['average_score']}/5.0
- **Questions Completed:** {performance['overall_performance']['total_questions']}
- **Performance Range:** {performance['overall_performance']['score_range']}

### Dimension Performance
"""
        
        for dim, score in performance['dimension_performance'].items():
            report += f"- **{dim.replace('_', ' ').title()}:** {score}/5.0\n"
        
        report += f"""
## Key Insights

"""
        for insight in insights['key_insights']:
            report += f"- {insight}\n"
        
        if config.include_improvement_suggestions:
            report += f"""
## Recommendations

"""
            for rec in insights['recommendations']:
                report += f"- {rec}\n"
        
        if config.include_detailed_scores:
            report += f"""
## Detailed Question Analysis

"""
            for qa in data['questions_and_answers']:
                report += f"""
### Question {qa['question_number']}: {qa['topic']} ({qa['difficulty']})

**Score:** {qa['score']}/5.0

**Strengths:**
"""
                for strength in qa['feedback']['strengths']:
                    report += f"- {strength}\n"
                
                if qa['feedback']['improvements']:
                    report += "\n**Areas for Improvement:**\n"
                    for improvement in qa['feedback']['improvements']:
                        report += f"- {improvement}\n"
        
        return report.strip()
    
    async def _save_report(self,
                          content: str,
                          config: ReportConfiguration,
                          interview_data: InterviewData) -> Path:
        """Save the report to file and return the file path."""
        
        # Generate filename
        session_info = interview_data.session_info
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session_info.get('role', 'interview')}_{config.report_type}_{timestamp}.{config.format}"
        
        file_path = self.output_dir / filename
        
        # Save based on format
        if config.format in ['markdown', 'md', 'html', 'txt']:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif config.format == 'json':
            # Convert to JSON format
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': config.report_type,
                    'session_id': session_info.get('session_id')
                },
                'content': content,
                'interview_data': asdict(interview_data)
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        elif config.format == 'pdf':
            # For PDF generation, you'd use libraries like reportlab or weasyprint
            # For now, save as markdown and note PDF conversion needed
            md_path = file_path.with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"PDF requested but saved as markdown: {md_path}")
            return md_path
        
        logger.info(f"Report saved: {file_path}")
        return file_path
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation."""
        payload = {
            "model": ollama_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 400
            }
        }
        
        response = await self.ollama_client.post("/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    async def close(self):
        """Clean up resources."""
        await self.ollama_client.aclose()

# Utility functions for easy report generation
async def generate_candidate_report(session_data: Dict,
                                  scores: List[ScoreResult], 
                                  feedback: List[FeedbackResult],
                                  format: str = "markdown") -> Dict:
    """Convenience function for generating candidate reports."""
    generator = ReportGenerator()
    
    config = ReportConfiguration(
        report_type="candidate",
        format=format,
        include_detailed_scores=True,
        include_improvement_suggestions=True,
        include_charts=True,
        include_raw_answers=False
    )
    
    try:
        result = await generator.generate_interview_report(
            session_data, scores, feedback, config
        )
        return result
    finally:
        await generator.close()

# CLI interface for testing
async def main():
    """CLI interface for testing report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Report Generator for AI Interview Coach")
    parser.add_argument("--type", choices=["candidate", "interviewer", "summary"], 
                       default="candidate", help="Report type")
    parser.add_argument("--format", choices=["markdown", "json", "html", "pdf"], 
                       default="markdown", help="Output format")
    parser.add_argument("--sample", action="store_true", 
                       help="Generate sample report with mock data")
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    if args.sample:
        # Create sample data for testing
        session_data = {
            'session_id': 'test_session_001',
            'candidate_name': 'John Doe',
            'role': 'software_engineer',
            'level': 'mid',
            'start_time': datetime.now(),
            'total_questions': 3
        }
        
        # This would normally come from actual scoring and feedback
        print("Sample report generation not implemented in this demo.")
        print("In a real system, you would pass actual ScoreResult and FeedbackResult objects.")
        return
    
    print("Report generator initialized successfully!")
    print(f"Templates directory: {Path('templates').resolve()}")
    print(f"Output directory: {Path('reports').resolve()}")

if __name__ == "__main__":
    # Add numpy import for trajectory chart
    import numpy as np
    asyncio.run(main())