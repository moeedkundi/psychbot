# AI Interview Coach

A modern AI-powered technical interview coaching system built with Chainlit and FastAPI. Practice technical interviews, receive real-time feedback, and get comprehensive performance reports.

## Features

- **AI-Powered Interviews**: Uses local Llama models via Ollama for realistic interview conversations
- **RAG-Enhanced Questions**: 170+ questions with FAISS vector database for contextual retrieval
- **Real-time Feedback**: Instant evaluation and scoring across multiple competencies
- **Interactive UI**: Modern chat interface with clickable role selection buttons
- **Flexible Sessions**: Choose 2-50 questions with early quit functionality
- **Comprehensive Reports**: Detailed analysis in JSON, Markdown, and HTML formats
- **Role-Specific**: Data Scientists (Junior/Mid/Senior), Software Engineers, GenAI Engineers
- **Random Questions**: Varied question selection for better practice experience

## Architecture

- **Frontend**: Chainlit chat interface with interactive action buttons
- **Backend**: FastAPI with async/await for high performance
- **AI Engine**: Local Llama 3.2 model via Ollama for privacy and control
- **Vector Database**: FAISS with Ollama embeddings for semantic search
- **Question Processing**: RAG system with 170+ curated technical questions
- **Configuration**: YAML contexts for AI behavior customization

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/moeedkundi/psychbot.git
   cd psychbot
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```
   
3. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Ollama configuration
   ```

4. **Start Ollama and pull the model:**
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

5. **Initialize the RAG database:**
   ```bash
   uv run python scripts/rag_retriever.py --action load
   ```

6. **Start the services:**
   ```bash
   # Terminal 1: Start backend
   uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
   
   # Terminal 2: Start frontend
   uv run chainlit run app.py --port 8080
   ```

7. **Access the application:**
   - Interview Interface: http://localhost:8080
   - API Documentation: http://localhost:8000/docs

## Project Structure

```
psychbot/
├── app.py                          # Chainlit frontend
├── backend/
│   ├── main.py                     # FastAPI backend server
│   └── reports/                    # Generated report storage
├── scripts/                        # Core AI interview logic
│   ├── ask_question.py             # Question orchestration
│   ├── comprehensive_evaluator.py  # AI answer evaluation
│   ├── generate_feedback.py        # Feedback generation
│   ├── generate_report.py          # Report generation
│   ├── rag_retriever.py           # RAG system with FAISS
│   └── score_answer.py            # Answer scoring
├── contexts/                       # AI behavior configurations
│   ├── interviewer.context.yaml
│   ├── feedback_generator.context.yaml
│   └── scorer.context.yaml
├── docs/                          # Knowledge base
│   ├── concepts/                  # Technical explanations
│   ├── interview_questions/       # 170+ curated questions
│   └── job_rubrics/              # Evaluation criteria
├── data/                          # FAISS vector database
├── reports/                       # Generated interview reports
└── tests/                         # Test suite
```

## Usage

### Starting an Interview

1. **Navigate to http://localhost:8080**
2. **Click your preferred role button:**
   - Data Scientist (Junior/Mid/Senior)
   - Software Engineer (Mid/Senior)  
   - GenAI Engineer (Mid)

3. **Enter number of questions:** Choose between 2-50 questions

4. **Interactive Interview:**
   - Answer questions in natural language
   - Type "quit" anytime to end early
   - Receive real-time AI evaluation and scoring

5. **Generate Reports:**
   - Type "generate report" for interactive format selection
   - Get detailed performance analysis and improvement suggestions

### API Usage

The system provides REST API endpoints:

```python
import httpx

# Start interview session
async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8000/api/interview/start", json={
        "role": "data_scientist",
        "level": "mid",
        "total_questions": 5
    })
    session_id = response.json()["session_id"]

    # Get first question
    response = await client.get(f"http://localhost:8000/api/interview/{session_id}/question")
    question_data = response.json()

    # Submit answer
    response = await client.post("http://localhost:8000/api/interview/answer", json={
        "session_id": session_id,
        "answer": "Your technical answer here"
    })
    feedback = response.json()
```

## Configuration

Configure via `.env` file:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# FAISS Vector Database
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=interview_knowledge

# API Server
API_HOST=0.0.0.0
API_PORT=8000
CHAINLIT_PORT=8080
```

## Available Question Types

- **Data Science**: Statistics, ML algorithms, feature engineering, model evaluation, SQL, business communication (170+ questions)
- **Software Engineering**: Algorithms, system design, coding challenges (75+ questions)
- **GenAI Engineering**: LLM architectures, RAG systems, prompt engineering (25+ questions)

## Commands During Interview

- **Answer questions**: Type your response naturally
- **Quit interview**: Type "quit", "exit", "stop", or "end interview"
- **Generate report**: Type "generate report" (after completing or quitting)
- **Start new session**: Type "yes" after completing an interview
- **Report formats**: "json", "markdown", "html" for specific formats

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code  
uv run ruff check .
```

### Managing Questions

```bash
# Reload RAG database after adding questions
uv run python scripts/rag_retriever.py --action load --force-reload

# Check database statistics
uv run python scripts/rag_retriever.py --action stats
```

## Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**FAISS Database Issues:**
```bash
# Rebuild vector database
uv run python scripts/rag_retriever.py --action load --force-reload
```

**Port Conflicts:**
```bash
# Check if ports are in use
netstat -an | findstr :8000
netstat -an | findstr :8080
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.

---

**Built for better technical interviews**