# AI Concierge for Small-Business AI Adoption

A sophisticated AI concierge system designed to help small businesses explore and adopt AI technologies. The concierge can answer questions about AI, schedule demo calls, and maintain a to-do list, all while continuously improving through self-grading and feedback mechanisms.

## Features

- **AI Knowledge Base**: Answers questions about AI technologies with citations
- **Task Management**: Schedules demos and maintains to-do lists
- **Self-Grading**: Evaluates the relevance and coverage of retrieved information
- **Self-Reflection**: Adapts behavior based on user feedback
- **Secure API**: JWT authentication and rate limiting
- **Optional Voice Interface**: Process audio input and generate audio responses

## Architecture

This project uses the LangGraph framework from the LangChain ecosystem (Option A from the requirements) to create a sophisticated conversation agent. The core components include:

- **Vector Search RAG**: Retrieves relevant information from a knowledge base
- **Task Manager**: Handles scheduling and to-do list management
- **Self-Grading System**: Evaluates the quality of retrieved documents
- **Adaptive Learning**: Modifies behavior based on user feedback
- **FastAPI Backend**: Provides secure API endpoints with authentication

### LLM Options

The application is designed to work with different LLM providers:

1. **OpenAI (Default if API key provided)**: 
   - Requires: API key and payment
   - Advantages: High performance and reliability

2. **Hugging Face Models (Free alternatives)**:
   - Requires: Hugging Face API token (free to create)
   - Options:
     - Llama models from Meta (requires HF Pro for API access)
     - Mistral models (open access)
     - Many other open-source models

3. **Local deployment options** (advanced):
   - For local Llama deployment (requires C++ compiler):
     ```bash
     # Install required dependencies
     sudo apt-get install build-essential
     # Then install llama-cpp-python
     pip install llama-cpp-python
     ```

To customize the LLM:

```python
# Example to use a different HuggingFace model
from langchain_community.llms import HuggingFaceEndpoint
from app.services.concierge import ConciergeService

hf_model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1
)
concierge = ConciergeService(llm=hf_model)
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key (for LLM access)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-concierge.git
   cd ai-concierge
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   # Option 1: Use OpenAI (requires payment)
   OPENAI_API_KEY=your_api_key_here
   
   # Option 2: Use open-source models
   # Leave OPENAI_API_KEY empty to use Hugging Face models
   # OPENAI_API_KEY=
   ```

### Running with Docker

The easiest way to run the application is with Docker Compose:

```bash
docker-compose up -d
```

This will start:
- The AI Concierge API on port 8000
- ChromaDB vector database on port 8001

### Running Locally

To run the application without Docker:

```bash
uvicorn app.main:app --reload
```

## API Usage

### Authentication

1. Register a new user:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/register" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "securepassword"}'
   ```

2. Login to get an access token:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user@example.com&password=securepassword"
   ```

3. Use the token for authenticated requests:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/concierge/chat" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is few-shot learning?"}'
   ```

### Example Interactions

#### Asking about AI technology:

```bash
curl -X POST "http://localhost:8000/api/v1/concierge/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is vector search?"}'
```

Expected response:
```json
{
  "response": "Vector search is a method for finding similar items in a large dataset by representing items as vectors in a high-dimensional space and measuring the distance between them. It enables AI applications like semantic search, recommendation systems, and anomaly detection by finding conceptually similar items rather than just matching keywords. [Source: ML Basics, Chapter 7]"
}
```

#### Scheduling a demo:

```bash
curl -X POST "http://localhost:8000/api/v1/concierge/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Schedule an AI demo for Thursday at 3 PM"}'
```

Expected response:
```json
{
  "response": "✓ Demo scheduled for Thursday at 3:00 PM."
}
```

#### Providing feedback:

```bash
curl -X POST "http://localhost:8000/api/v1/concierge/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "/good_answer"}'
```

Expected response:
```json
{
  "response": "✓ Thank you for the positive feedback! I'll maintain this style."
}
```

## Self-Grading Mechanism

The concierge uses a sophisticated self-grading mechanism to evaluate the quality of retrieved information:

1. When a user asks a question, the system retrieves relevant documents from the knowledge base.
2. These documents are evaluated on two criteria:
   - **Factual Relevance**: How relevant the documents are to the question (0-1)
   - **Answer Coverage**: How completely the documents can answer the question (0-1)
3. If either score falls below 0.6, the system attempts to refine the search query.
4. If the scores remain low after refinement, the system acknowledges its knowledge gap.

This approach ensures the concierge provides high-quality, relevant information and is transparent about its limitations.

### Threshold Justification

The threshold of 0.6 was chosen after extensive testing to balance:
- Precision: Ensuring responses are relevant and accurate
- Recall: Maximizing the range of questions that can be answered
- User Experience: Minimizing "I don't know" responses while maintaining quality

## Self-Reflection Algorithm

The concierge implements an adaptive learning mechanism based on user feedback:

1. Users can provide explicit feedback with `/good_answer` or `/bad_answer` commands.
2. A session-specific feedback score is maintained:
   - `/good_answer`: +1 to the score
   - `/bad_answer`: -1 to the score
3. The score decays by a factor of 0.5 in each conversation turn to prioritize recent feedback.
4. The system modifies its behavior based on the cumulative score:
   - Negative score: Be more concise and cite sources explicitly
   - Positive score: Maintain the current style

### Decay Mechanism

The decay factor (0.5) ensures that:
- Recent feedback has more impact than older feedback
- The system can recover from occasional negative feedback
- Long-term patterns still influence behavior

## Testing

Run the test suite with:

```bash
pytest
```

The test suite includes:
- Unit tests for core functions
- Integration tests for API endpoints
- Tests for self-grading and self-reflection mechanisms

## Project Structure

```
project/
├── app/
│   ├── api/                 # API endpoints
│   ├── core/                # Core settings and utilities
│   ├── models/              # Data models
│   ├── services/            # Business logic
│   │   ├── tools/           # Agent tools
│   └── utils/               # Utility functions
├── data/                    # Knowledge base
├── tests/                   # Test suite
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-container setup
└── requirements.txt         # Dependencies
```

## License

[MIT License](LICENSE)