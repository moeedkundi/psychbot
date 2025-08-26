#!/usr/bin/env python3
"""
Chainlit Frontend for Optimized AI Interview Coach
Modern chat interface that connects to the optimized backend.
"""

import chainlit as cl
try:
    from chainlit.input_widget import Select
except ImportError:
    try:
        from chainlit import Select
    except ImportError:
        Select = None
import httpx
import asyncio
import json
from typing import Dict, Any, Optional
import time

# Configuration
BACKEND_URL = "http://localhost:8000"
TIMEOUT = 180.0  # 3 minutes to handle long AI evaluation times

# Session state
current_session: Dict[str, Any] = {}

@cl.on_chat_start
async def start():
    """Initialize the chat session with interactive role selection."""
    
    # Create interactive action buttons for role selection (without question count)
    actions = [
        cl.Action(
            name="data_scientist_junior",
            payload={"value": "data_scientist,junior"}, 
            label="🔬 Data Scientist (Junior)",
            tooltip="🔬 Data Scientist - Junior Level"
        ),
        cl.Action(
            name="data_scientist_mid",
            payload={"value": "data_scientist,mid"},
            label="🔬 Data Scientist (Mid)",
            tooltip="🔬 Data Scientist - Mid Level"
        ),
        cl.Action(
            name="data_scientist_senior", 
            payload={"value": "data_scientist,senior"},
            label="🔬 Data Scientist (Senior)",
            tooltip="🔬 Data Scientist - Senior Level"
        ),
        cl.Action(
            name="software_engineer_mid",
            payload={"value": "software_engineer,mid"},
            label="💻 Software Engineer (Mid)",
            tooltip="💻 Software Engineer - Mid Level"
        ),
        cl.Action(
            name="software_engineer_senior",
            payload={"value": "software_engineer,senior"},
            label="💻 Software Engineer (Senior)",
            tooltip="💻 Software Engineer - Senior Level"
        ),
        cl.Action(
            name="genai_engineer_mid",
            payload={"value": "genai_engineer,mid"},
            label="🤖 GenAI Engineer (Mid)",
            tooltip="🤖 GenAI Engineer - Mid Level"
        )
    ]
    
    # Enhanced welcome message with interactive selection
    await cl.Message(
        content="# 🚀 Welcome to AI Interview Coach!\n\n"
               "### 🎯 Your Intelligent Technical Interview Partner\n\n"
               "Ready to practice your technical interview skills? Choose your role and level below:\n\n"
               "**📊 Our Database:**\n"
               "- 🔬 **Data Scientist**: 170+ questions across all levels\n"
               "- 💻 **Software Engineer**: 50+ questions (Mid/Senior)\n" 
               "- 🤖 **GenAI Engineer**: 25+ questions (Mid level)\n\n"
               "**✨ What you'll get:**\n"
               "- ⚡ Real-time AI evaluation\n"
               "- 📊 Multi-dimensional scoring\n"
               "- 💡 Detailed feedback & improvement tips\n"
               "- 📄 Comprehensive performance reports\n\n"
               "**👇 Click a button below to start your interview:**"
    ).send()
    
    # Use AskActionMessage for interactive selection
    res = await cl.AskActionMessage(
        content="Please select your interview configuration:",
        actions=actions,
        timeout=300  # 5 minute timeout
    ).send()
    
    if res and hasattr(res, 'get'):
        # Parse the action value (format: "role,level")
        payload_value = res.get("payload", {}).get("value", "")
        role_config = payload_value.split(',')
        if len(role_config) == 2:
            role, level = role_config
            
            # Ask for number of questions with validation
            question_count = await ask_for_question_count(role, level)
            if question_count:
                # Start interview with selected configuration
                await cl.Message(content=f"🚀 **Starting {role.replace('_', ' ').title()} - {level.title()} Interview**\n"
                                         f"📝 **Questions:** {question_count}\n"
                                         f"⏱️ **Preparing your first question...**").send()
                
                # Call start_interview_with_settings directly with proper parameters
                await start_interview_with_settings(role, level, question_count, "Candidate")
        else:
            await cl.Message(content="❌ Invalid selection. Please restart to try again.").send()
    else:
        await cl.Message(content="❌ No selection made. Please restart to try again.").send()

async def ask_for_question_count(role: str, level: str) -> Optional[int]:
    """Ask user for number of questions with validation."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Ask for question count
            question_msg = await cl.AskUserMessage(
                content=f"📝 **{role.replace('_', ' ').title()} - {level.title()}** selected!\n\n"
                        f"How many questions would you like? (2-20 questions recommended)\n"
                        f"💡 Enter a number between 2 and 50:",
                timeout=60
            ).send()
            
            if question_msg:
                # Handle different response formats - try multiple approaches
                user_input = None
                try:
                    if hasattr(question_msg, 'content'):
                        user_input = question_msg.content.strip()
                    elif isinstance(question_msg, dict):
                        # Try common dict keys
                        for key in ['content', 'output', 'message', 'text', 'value']:
                            if key in question_msg:
                                user_input = str(question_msg[key]).strip()
                                break
                        # If still no user_input, show debug info
                        if not user_input:
                            await cl.Message(content=f"Debug: Dict keys: {list(question_msg.keys())}").send()
                            await cl.Message(content=f"Debug: Full dict: {question_msg}").send()
                            user_input = str(question_msg).strip()
                    else:
                        user_input = str(question_msg).strip()
                except Exception as debug_error:
                    await cl.Message(content=f"Debug error: {debug_error}").send()
                    user_input = str(question_msg).strip()
                
                # Validate numeric input
                try:
                    question_count = int(user_input)
                    if 2 <= question_count <= 50:
                        return question_count
                    else:
                        if attempt < max_attempts - 1:
                            await cl.Message(content="⚠️ Please enter a number between 2 and 50. Try again:").send()
                        continue
                except ValueError:
                    if attempt < max_attempts - 1:
                        await cl.Message(content="❌ Please enter a valid number (not text). Try again:").send()
                    continue
            else:
                break
                
        except Exception as e:
            await cl.Message(content=f"❌ Error getting input: {str(e)}").send()
            break
    
    # If we get here, validation failed after max attempts
    await cl.Message(content="❌ Too many invalid attempts. Please restart to try again.").send()
    return None

async def handle_quit_interview():
    """Handle user request to quit the interview."""
    global current_session
    
    if not current_session.get('session_id'):
        await cl.Message(content="⚠️ No active interview session to quit.").send()
        return
    
    # Get session summary before quitting
    session_id = current_session['session_id']
    questions_answered = current_session.get('question_count', 0) 
    total_questions = current_session.get('total_questions', 5)
    
    # Notify backend about early termination by finalizing the session
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Force session completion by getting the final question (this moves it to session_results)
            response = await client.get(f"{BACKEND_URL}/api/interview/{session_id}/question")
            # This should trigger completion logic and move session to session_results
    except Exception as e:
        # If backend call fails, continue with local termination
        pass
    
    # Mark session as terminated but keep data for potential report generation
    current_session['status'] = 'terminated'
    current_session['waiting_for_answer'] = False
    
    await cl.Message(
        content=f"🛑 **Interview Terminated**\n\n"
               f"📊 **Session Summary:**\n"
               f"- **Session ID**: `{session_id}`\n"
               f"- **Questions Answered**: {questions_answered}/{total_questions}\n"
               f"- **Status**: Terminated by user\n\n"
               f"Thank you for your time! To start a new interview, type **yes** to return to the role selection screen.\n\n"
               f"💡 **Tip**: If you answered any questions, you can still generate a partial report by typing `generate report`"
    ).send()

async def show_welcome_screen():
    """Display the welcome screen with role selection buttons."""
    global current_session
    
    # Clear any existing session
    current_session.clear()
    
    # Create interactive action buttons for role selection (copied from start() function)
    actions = [
        cl.Action(
            name="data_scientist_junior",
            payload={"value": "data_scientist,junior"}, 
            label="🔬 Data Scientist (Junior)",
            tooltip="🔬 Data Scientist - Junior Level"
        ),
        cl.Action(
            name="data_scientist_mid",
            payload={"value": "data_scientist,mid"},
            label="🔬 Data Scientist (Mid)",
            tooltip="🔬 Data Scientist - Mid Level"
        ),
        cl.Action(
            name="data_scientist_senior", 
            payload={"value": "data_scientist,senior"},
            label="🔬 Data Scientist (Senior)",
            tooltip="🔬 Data Scientist - Senior Level"
        ),
        cl.Action(
            name="software_engineer_mid",
            payload={"value": "software_engineer,mid"},
            label="💻 Software Engineer (Mid)",
            tooltip="💻 Software Engineer - Mid Level"
        ),
        cl.Action(
            name="software_engineer_senior",
            payload={"value": "software_engineer,senior"},
            label="💻 Software Engineer (Senior)",
            tooltip="💻 Software Engineer - Senior Level"
        ),
        cl.Action(
            name="genai_engineer_mid",
            payload={"value": "genai_engineer,mid"},
            label="🤖 GenAI Engineer (Mid)",
            tooltip="🤖 GenAI Engineer - Mid Level"
        )
    ]
    
    # Enhanced welcome message with interactive selection
    await cl.Message(
        content="# 🚀 Welcome to AI Interview Coach!\n\n"
               "### 🎯 Your Intelligent Technical Interview Partner\n\n"
               "Ready to practice your technical interview skills? Choose your role and level below:\n\n"
               "**📊 Our Database:**\n"
               "- 🔬 **Data Scientist**: 170+ questions across all levels\n"
               "- 💻 **Software Engineer**: 50+ questions (Mid/Senior)\n" 
               "- 🤖 **GenAI Engineer**: 25+ questions (Mid level)\n\n"
               "**✨ What you'll get:**\n"
               "- ⚡ Real-time AI evaluation\n"
               "- 📊 Multi-dimensional scoring\n"
               "- 💡 Detailed feedback & improvement tips\n"
               "- 📄 Comprehensive performance reports\n\n"
               "**👇 Click a button below to start your interview:**"
    ).send()
    
    # Use AskActionMessage for interactive selection
    res = await cl.AskActionMessage(
        content="Please select your interview configuration:",
        actions=actions,
        timeout=300  # 5 minute timeout
    ).send()
    
    if res and hasattr(res, 'get'):
        # Parse the action value (format: "role,level")
        payload_value = res.get("payload", {}).get("value", "")
        role_config = payload_value.split(',')
        if len(role_config) == 2:
            role, level = role_config
            
            # Ask for number of questions with validation
            question_count = await ask_for_question_count(role, level)
            if question_count:
                # Start interview with selected configuration
                await cl.Message(content=f"🚀 **Starting {role.replace('_', ' ').title()} - {level.title()} Interview**\n"
                                         f"📝 **Questions:** {question_count}\n"
                                         f"⏱️ **Preparing your first question...**").send()
                
                # Call start_interview_with_settings directly with proper parameters
                await start_interview_with_settings(role, level, question_count, "Candidate")
        else:
            await cl.Message(content="❌ Invalid selection. Please restart to try again.").send()
    else:
        await cl.Message(content="❌ No selection made. Please restart to try again.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    global current_session
    
    user_input = message.content.strip()
    
    try:
        # Check if we need to start an interview
        if not current_session.get('session_id'):
            await handle_interview_start(user_input)
            return  # Don't continue processing after starting interview
        
        # Check for report generation command (works even after interview completion)
        elif user_input.lower() in ['generate report', 'report']:
            await generate_report()
            return
        
        # Check for report generation with specific format
        elif user_input.lower().startswith('generate report ') or user_input.lower().startswith('report '):
            format_part = user_input.lower().split(' ', 2)[-1]  # Get the format part
            if format_part in ['markdown', 'json', 'html']:
                await generate_report_with_format(format_part)
            elif format_part == 'pdf':
                await cl.Message(content="⚠️ PDF format is not fully supported yet. Generating markdown report instead.").send()
                await generate_report_with_format('markdown')
            else:
                await generate_report()
            return
        
        # Check for just format name (quick generation)
        elif user_input.lower() in ['markdown', 'json', 'html']:
            await generate_report_with_format(user_input.lower())
            return
        
        # Check for quit commands
        elif user_input.lower() in ['quit', 'exit', 'stop', 'end interview', 'quit interview']:
            await handle_quit_interview()
            return
        
        # Check for restart/new interview command
        elif user_input.lower() in ['yes', 'restart', 'new interview', 'start over']:
            await show_welcome_screen()
            return
        
        # Check if we have an active question (and session is not terminated)
        elif current_session.get('waiting_for_answer') and current_session.get('status') != 'terminated':
            await handle_answer_submission(user_input)
        
        # Get next question (only if we have an active session that's not terminated)
        elif current_session.get('session_id') and current_session.get('status') != 'terminated':
            await get_next_question()
        
        # Handle input when no active session - try to start interview
        else:
            await handle_interview_start(user_input)
    
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error**: {str(e)}\n\nPlease try again or restart the interview."
        ).send()

async def start_interview_with_settings(role: str, level: str, total_questions: int, name: str = "Candidate"):
    """Start interview with pre-selected settings from UI."""
    global current_session
    
    # Show loading
    loading_msg = await cl.Message(content="🔄 Starting your interview session...").send()
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            
            # Check backend health
            health_response = await client.get(f"{BACKEND_URL}/health")
            if health_response.status_code != 200:
                raise Exception("Backend is not available")
            
            # Start interview
            start_time = time.time()
            interview_data = {
                "role": role,
                "level": level,
                "candidate_name": name,
                "total_questions": total_questions
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/interview/start", 
                json=interview_data
            )
            response.raise_for_status()
            
            result = response.json()
            start_duration = time.time() - start_time
            
            # Store session info
            current_session = {
                'session_id': result['session_id'],
                'candidate_name': name,
                'role': role,
                'level': level,
                'question_count': 0,
                'total_questions': total_questions,
                'waiting_for_answer': False
            }
            
            await loading_msg.remove()
            
            # Welcome message
            await cl.Message(
                content=f" **Interview Started!**\n\n"
                       f"👤 **Candidate**: {name}\n"
                       f" **Role**: {role.replace('_', ' ').title()}\n"
                       f" **Level**: {level.title()}\n"
                       f" **Setup Time**: {start_duration:.2f}s\n"
                       f" **Total Questions**: {total_questions}\n\n"
                       f"**Session ID**: `{result['session_id']}`\n\n"
                       f"{result['welcome_message']}\n\n"
                       f"Ready? Let's get your first question! 🚀"
            ).send()
            
            # Get first question
            await get_next_question()
            
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f" **Failed to start interview**: {str(e)}\n\n"
                   "Please check that the backend is running on port 8000 and try again."
        ).send()

async def handle_interview_start(user_input: str):
    """Parse user input and start interview session."""
    global current_session
    
    # Parse input
    parts = [p.strip() for p in user_input.split(',')]
    
    # Handle numbered role format (e.g., "1, 5" or "2, 7")
    if len(parts) >= 2:
        try:
            role_num = int(parts[0])
            total_questions = int(parts[1])
            
            # Map role numbers to role names
            role_mapping = {
                1: "software_engineer",
                2: "data_scientist", 
                3: "genai_engineer"
            }
            
            if role_num in role_mapping and 2 <= total_questions <= 50:
                role = role_mapping[role_num]
                level = "mid"  # Currently only mid level available
                await start_interview_with_settings(role, level, total_questions, "Candidate")
                return
            elif role_num in role_mapping:
                # Valid role but invalid question count
                await cl.Message(
                    content=f"⚠️ Invalid question count: **{total_questions}**\n\n"
                           f"Please choose number of questions.\n\n"
                           f"Example: `{role_num}, 5` for {role_mapping[role_num].replace('_', ' ').title()} with 5 questions"
                ).send()
                return
            elif 2 <= total_questions <= 50:
                # Valid question count but invalid role
                await cl.Message(
                    content=f"⚠️ Invalid role number: **{role_num}**\n\n"
                           f"Please choose from:\n"
                           f"1️⃣ Software Engineer\n2️⃣ Data Scientist\n3️⃣ GenAI Engineer\n\n"
                           f"Example: `2, {total_questions}` for Data Scientist with {total_questions} questions"
                ).send()
                return
        except ValueError:
            pass  # Not numbered format, continue with old parsing
    
    if len(parts) < 2:
        await cl.Message(
            content="⚠️ Please provide role and question count.\n\n"
                   "**Examples:**\n"
                   "• `1, 5` - Software Engineer, 5 questions\n"
                   "• `2, 7` - Data Scientist, 7 questions\n"
                   "• `3, 4` - GenAI Engineer, 4 questions\n\n"
                   "**Or use old format:**\n"
                   "• `software_engineer, mid, 5`"
        ).send()
        return
    
    role = parts[0].lower().replace(' ', '_')
    level = parts[1].lower()
    
    # Parse total_questions (3rd parameter, optional)
    total_questions = 5  # default
    if len(parts) > 2:
        try:
            # Try to parse as number for total_questions
            potential_questions = int(parts[2].strip())
            if 3 <= potential_questions <= 10:
                total_questions = potential_questions
                name = parts[3] if len(parts) > 3 else "Candidate"
            else:
                # If not a valid number, treat as name
                name = parts[2]
        except ValueError:
            # If can't parse as number, treat as name
            name = parts[2]
    else:
        name = "Candidate"
    
    # Show loading
    loading_msg = await cl.Message(content="🔄 Starting your interview session...").send()
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            
            # Check backend health
            health_response = await client.get(f"{BACKEND_URL}/health")
            if health_response.status_code != 200:
                raise Exception("Backend is not available")
            
            # Start interview
            start_time = time.time()
            interview_data = {
                "role": role,
                "level": level,
                "candidate_name": name,
                "total_questions": total_questions
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/interview/start", 
                json=interview_data
            )
            response.raise_for_status()
            
            result = response.json()
            start_duration = time.time() - start_time
            
            # Store session info
            current_session = {
                'session_id': result['session_id'],
                'candidate_name': name,
                'role': role,
                'level': level,
                'question_count': 0,
                'total_questions': total_questions,
                'waiting_for_answer': False
            }
            
            await loading_msg.remove()
            
            # Welcome message
            await cl.Message(
                content=f" **Interview Started!**\n\n"
                       f"👤 **Candidate**: {name}\n"
                       f" **Role**: {role.replace('_', ' ').title()}\n"
                       f" **Level**: {level.title()}\n"
                       f" **Setup Time**: {start_duration:.2f}s\n"
                       f" **Total Questions**: {total_questions}\n\n"
                       f"**Session ID**: `{result['session_id']}`\n\n"
                       f"{result['welcome_message']}\n\n"
                       f"Ready? Let's get your first question! 🚀"
            ).send()
            
            # Get first question
            await get_next_question()
            
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f" **Failed to start interview**: {str(e)}\n\n"
                   "Please check that the backend is running on port 8000 and try again."
        ).send()

async def get_next_question():
    """Get the next question from the backend."""
    global current_session
    
    if not current_session.get('session_id'):
        return
    
    loading_msg = await cl.Message(content=" Getting your next question...").send()
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            start_time = time.time()
            
            response = await client.get(
                f"{BACKEND_URL}/api/interview/{current_session['session_id']}/question"
            )
            response.raise_for_status()
            
            result = response.json()
            question_duration = time.time() - start_time
            
            await loading_msg.remove()
            
            if result['status'] == 'active':
                current_session['current_question'] = result
                current_session['waiting_for_answer'] = True
                current_session['question_count'] = result['question_number']
                
                # Display question
                await cl.Message(
                    content=f"❓ **Question {result['question_number']}/{result['total_questions']}**\n\n"
                           f"**Topic**: {result['metadata']['topic'].replace('_', ' ').replace('**', '').title()}\n"
                           f"**Difficulty**: {result['metadata']['difficulty'].replace('**', '').title()}\n"
                           f"⚡ **Retrieved in**: {question_duration:.2f}s\n\n"
                           f"---\n\n"
                           f"**{result['question'].replace('**', '')}**\n\n"
                           f"---\n\n"
                           f"💡 *Type your answer, or type 'quit' to end the interview*\n\n"
                ).send()
                
            elif result['status'] == 'complete':
                current_session['waiting_for_answer'] = False
                
                # Safely access session summary with fallbacks
                session_summary = result.get('session_summary', {})
                total_questions = session_summary.get('total_questions', current_session.get('total_questions', 5))
                questions_completed = session_summary.get('questions_completed', current_session.get('question_count', 0))
                avg_score = session_summary.get('average_score', 0.0)
                total_time = session_summary.get('total_time', 'Unknown')
                
                await cl.Message(
                    content=f"🎉 **Interview Complete!**\n\n"
                           f"Congratulations! You've completed all {total_questions} questions.\n\n"
                           f"📊 **Your Performance**:\n"
                           f"- **Questions Answered**: {questions_completed}\n"
                           f"- **Average Score**: {avg_score:.1f}/5.0\n"
                           f"- **Total Time**: {total_time}\n\n"
                           f"Would you like me to generate a detailed report? Type **'generate report'** to create one!"
                ).send()
                
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f"❌ **Error getting question**: {str(e)}"
        ).send()

async def handle_answer_submission(user_answer: str):
    """Submit answer and get feedback."""
    global current_session
    
    if not current_session.get('current_question'):
        await cl.Message(content="⚠️ No active question to answer.").send()
        return
    
    # Show loading
    loading_msg = await cl.Message(content="🔍 Analyzing your answer...").send()
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            start_time = time.time()
            
            answer_data = {
                "session_id": current_session['session_id'],
                "answer": user_answer
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/interview/answer",
                json=answer_data
            )
            response.raise_for_status()
            
            result = response.json()
            eval_duration = time.time() - start_time
            
            await loading_msg.remove()
            
            # Display feedback
            feedback = result['feedback']
            score = result['score']
            
            # Create score bar
            score_value = score['overall_score']
            score_bar = "🟩" * int(score_value) + "⬜" * (5 - int(score_value))
            
            feedback_content = f"📊 **Answer Evaluation**\n\n"
            feedback_content += f"**Overall Score**: {score_value:.1f}/5.0 {score_bar}\n"
            feedback_content += f"⚡ **Evaluation Time**: {eval_duration:.2f}s\n\n"
            
            if 'category_scores' in score:
                feedback_content += "**Category Breakdown**:\n"
                for category, cat_score in score['category_scores'].items():
                    cat_bar = "🟩" * int(cat_score) + "⬜" * (5 - int(cat_score))
                    feedback_content += f"- **{category.replace('_', ' ').title()}**: {cat_score:.1f}/5.0 {cat_bar}\n"
                feedback_content += "\n"
            
            feedback_content += f"**Detailed Feedback**:\n{feedback['feedback']}\n\n"
            
            if feedback.get('follow_up_question'):
                feedback_content += f"🤔 **Follow-up**: {feedback['follow_up_question']}\n\n"
            
            feedback_content += "---\n\n"
            
            await cl.Message(content=feedback_content).send()
            
            # Check if interview continues
            if result.get('next_question'):
                if result['next_question']['status'] == 'active':
                    # Next question data is already provided in the response, use it directly
                    next_q = result['next_question']
                    current_session['current_question'] = next_q
                    current_session['waiting_for_answer'] = True
                    current_session['question_count'] = next_q.get('question_number', current_session.get('question_count', 0) + 1)
                    
                    # Display the next question directly with safe field access
                    question_num = next_q.get('question_number', current_session.get('question_count', 1))
                    total_questions = next_q.get('total_questions', current_session.get('total_questions', 5))
                    topic = next_q.get('metadata', {}).get('topic', 'General').replace('_', ' ').replace('**', '').title()
                    difficulty = next_q.get('metadata', {}).get('difficulty', 'Medium').replace('**', '').title()
                    question_text = next_q.get('question', 'Question not available').replace('**', '')
                    
                    await cl.Message(
                        content=f"❓ **Question {question_num}/{total_questions}**\n\n"
                               f"**Topic**: {topic}\n"
                               f"**Difficulty**: {difficulty}\n"
                               f"⚡ **Retrieved in**: 0.00s\n\n"
                               f"---\n\n"
                               f"**{question_text}**\n\n"
                               f"---\n\n"
                               f"💡 *Type your answer, or type 'quit' to end the interview*\n\n"
                    ).send()
                elif result['next_question']['status'] == 'complete':
                    current_session['waiting_for_answer'] = False
                    
                    # Safely access session summary with fallbacks
                    summary = result['next_question'].get('session_summary', {})
                    total_questions = summary.get('total_questions', current_session.get('total_questions', 5))
                    questions_completed = summary.get('questions_completed', current_session.get('question_count', 0))
                    avg_score = summary.get('average_score', 0.0)
                    total_time = summary.get('total_time', 'Unknown')
                    
                    await cl.Message(
                        content=f"🎉 **Interview Complete!**\n\n"
                               f"📊 **Final Results**:\n"
                               f"- **Questions**: {questions_completed}/{total_questions}\n"
                               f"- **Average Score**: {avg_score:.1f}/5.0\n"
                               f"- **Total Time**: {total_time}\n\n"
                               f"Type **'generate report'** to create a detailed report!"
                    ).send()
            else:
                current_session['waiting_for_answer'] = False
                await get_next_question()
                
    except Exception as e:
        await loading_msg.remove()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        await cl.Message(
            content=f"❌ **Error processing answer**: {error_msg}\n\nError type: {type(e).__name__}"
        ).send()
        # Also log the full error for debugging
        import traceback
        print(f"Answer processing error: {traceback.format_exc()}")

async def generate_report_with_format(format_type: str):
    """Generate interview report with specific format."""
    global current_session
    
    if not current_session.get('session_id'):
        await cl.Message(content="⚠️ No completed session found.").send()
        return
    
    loading_msg = await cl.Message(content=f"📄 Generating your detailed report in **{format_type.upper()}** format...").send()
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            start_time = time.time()
            
            report_data = {
                "session_id": current_session['session_id'],
                "report_type": "candidate",
                "format": format_type,
                "include_charts": True
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/report/generate",
                json=report_data
            )
            response.raise_for_status()
            
            result = response.json()
            report_duration = time.time() - start_time
            
            await loading_msg.remove()
            
            await cl.Message(
                content=f"✅ **Report Generated Successfully!**\n\n"
                       f"📄 **File**: `{result['report_file']}`\n"
                       f"⚡ **Generation Time**: {report_duration:.2f}s\n"
                       f"📊 **Report Type**: {result.get('type', 'candidate').title()}\n"
                       f"📝 **Format**: {result.get('format', 'markdown').upper()}\n\n"
                       f"Your comprehensive interview report has been generated and saved.\n"
                       f"It includes detailed feedback, performance analysis, and improvement suggestions.\n\n"
                       f"**Want to start another interview?** Just type: **yes**"
            ).send()
            
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f"❌ **Error generating report**: {str(e)}"
        ).send()

async def generate_report():
    """Generate interview report."""
    global current_session
    
    if not current_session.get('session_id'):
        await cl.Message(content="⚠️ No completed session found.").send()
        return
    
    # Ask user to choose format with direct commands
    await cl.Message(
        content="📄 **Choose Report Format:**\n\n"
               "Please type one of these commands:\n"
               "• `generate report markdown` - Easy to read text format\n"
               "• `generate report json` - Structured data format\n"
               "• `generate report html` - Web page format\n\n"
               "Or type `markdown` for quick default generation."
    ).send()
    
    return  # Exit here, let user choose format
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            start_time = time.time()
            
            report_data = {
                "session_id": current_session['session_id'],
                "report_type": "candidate",
                "format": chosen_format,
                "include_charts": True
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/report/generate",
                json=report_data
            )
            response.raise_for_status()
            
            result = response.json()
            report_duration = time.time() - start_time
            
            await loading_msg.remove()
            
            await cl.Message(
                content=f"✅ **Report Generated Successfully!**\n\n"
                       f"📄 **File**: `{result['report_file']}`\n"
                       f"⚡ **Generation Time**: {report_duration:.2f}s\n"
                       f"📊 **Report Type**: {result.get('type', 'candidate').title()}\n"
                       f"📝 **Format**: {result.get('format', 'markdown').upper()}\n\n"
                       f"Your comprehensive interview report has been generated and saved.\n"
                       f"It includes detailed feedback, performance analysis, and improvement suggestions.\n\n"
                       f"**Want to start another interview?** Just type: **yes**"
            ).send()
            
            # Reset session for new interview
            current_session.clear()
            
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f"❌ **Error generating report**: {str(e)}"
        ).send()

if __name__ == "__main__":
    # This will be run by: chainlit run app.py
    pass