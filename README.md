# AI Python 2

A FastAPI-based interview-prep assistant that ingests a candidate's resume, job description, and self-description, stores them as vector embeddings in MongoDB, and then answers questions using a LangChain agent with RAG + web search.

This project is designed to help a user ask natural-language questions about a specific interview or hiring context, such as:

- "How does my experience match this role?"
- "What skills are emphasized in the job description?"
- "What part of my background best supports my fit for this position?"
- "What is the company like, based on this job description and current public information?"

The assistant combines:

- vector similarity search over stored interview documents
- source-specific retrieval by document type
- a tool-using LLM workflow with Google Gemini
- optional web lookup via Tavily

---

## Overview

This service exposes two main API flows:

1. Ingest interview data
2. Chat with an agent grounded in that interview's source material

The application keeps a `MongoDB` collection named `Genai_resumeChecker.ragCollection` and stores chunks of text along with generated embeddings. Each stored chunk is tagged with:

- `interviewID`
- `sourceType` (`resume`, `jobDescription`, `selfDescription`)
- `text`
- `embedding`

During chat, the app restricts retrieval to the relevant `interviewID` and source type, then passes the matched text back into the LLM as tool output so the answer remains anchored in the candidate's actual materials.

---

## Key Features

- Resume, job description, and self-description ingestion
- Chunking and embedding generation for each document
- MongoDB vector search using `$vectorSearch`
- Separate retrieval tools for each source type
- Web search support for public/external information
- Session-aware chat memory using LangGraph in-memory checkpointer
- Internal API key validation for service-to-service access
- FastAPI endpoints for ingestion, chat, and health checks

---

## Tech Stack

- Python 3.14+
- FastAPI
- Pydantic
- MongoDB + PyMongo
- LangChain
- LangGraph
- Google Generative AI embeddings and chat model
- Tavily Search
- Uvicorn

---

## Project Structure

```text
Ai-python2/
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── uv.lock
├── src/
│   └── ai_python2/
│       ├── __init__.py
│       ├── Auth.py
│       ├── Controllers.py
│       ├── Models.py
│       ├── UtilFuncs.py
│       ├── flow.txt
│       └── fileStructue.txt
└── .venv/
```

Core files:

- `src/ai_python2/__init__.py` — app startup and route registration
- `src/ai_python2/Models.py` — request schemas for ingest and chat
- `src/ai_python2/UtilFuncs.py` — chunking, embeddings, and vector retrieval
- `src/ai_python2/Controllers.py` — main ingestion and chat logic
- `src/ai_python2/Auth.py` — internal API key verification

---

## Runtime Flow

### 1. Application startup

On app boot, the project:

- loads environment variables from `.env`
- establishes a MongoDB client
- creates an in-memory checkpointer for session-based chat state
- verifies the database connection with a `ping`

This is handled in `src/ai_python2/__init__.py`.

### 2. Ingestion

The `/ingest` route accepts a payload with:

- `interviewID`
- `userID`
- `resumeText`
- `jobDescription`
- `selfDescription`

The code validates the `interviewID`, splits each text block into chunks, generates embeddings, and inserts them into the MongoDB collection using a per-document `sourceType` discriminator.

### 3. Retrieval and chat

The `/chat` route accepts a message for a given interview and session, then:

- validates the `interviewID`
- creates a Google Gemini LLM
- defines tool functions for:
  - resume search
  - job description search
  - self-description search
  - web search via Tavily
- calls an agent that decides which tools to use before answering
- returns the final answer as the assistant response

This uses a LangGraph-style agent loop with conversation state scoped by `sessionID`.

---

## Data Models

### InterviewRequest

```python
class InterviewRequest(BaseModel):
    interviewID: str
    userID: str
    resumeText: str
    jobDescription: str
    selfDescription: str
```

### ChatRequest

```python
class ChatRequest(BaseModel):
    interviewID: str
    userID: str
    sessionID: str
    message: str
```

These models are defined in `src/ai_python2/Models.py` and used directly by FastAPI request validation.

---

## API Endpoints

### GET /

Returns a simple hello message.

```json
{
  "message": "Hello World"
}
```

### GET /health

Used for uptime and deployment checks.

```json
{
  "status": "ok"
}
```

### POST /ingest

Accepts interview source materials and stores them as embedded chunks.

Required auth header:

```http
X-Internal-Key: <your-shared-key>
```

Request body:

```json
{
  "interviewID": "64f7a3c1448b8b29e175f7d1",
  "userID": "user_123",
  "resumeText": "Senior software engineer with 6 years of experience in backend systems...",
  "jobDescription": "We are looking for a backend engineer with Python, FastAPI, and cloud experience...",
  "selfDescription": "I am a detail-oriented engineer with a strong focus on product quality..."
}
```

Success response:

```json
{
  "message": "ingested successfully"
}
```

### POST /chat

Sends a user message for a specific interview session and returns the model's response.

Required auth header:

```http
X-Internal-Key: <your-shared-key>
```

Request body:

```json
{
  "interviewID": "64f7a3c1448b8b29e175f7d1",
  "userID": "user_123",
  "sessionID": "session-001",
  "message": "Which parts of my background match this backend role the best?"
}
```

Success response:

```json
{
  "message": "response given successfully",
  "response": "Your resume highlights ..."
}
```

---

## Environment Variables

The app expects environment variables such as:

```env
MONGO_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
INTERNAL_API_KEY="your-shared-service-key"
GOOGLE_API_KEY="your-google-gemini-key"
TAVILY_API_KEY="your-tavily-key"
```

The project uses `python-dotenv` to load values from a `.env` file at runtime.

> The internal key protects service-to-service endpoints. If the header is missing or invalid, FastAPI returns HTTP 401.

---

## Setup

### 1. Clone the project

```bash
git clone <repository-url>
cd Ai-python2
```

### 2. Create and activate a virtual environment

If you are using `uv`:

```bash
uv sync
```

Then run:

```bash
uv run uvicorn ai_python2:app --reload
```

If you are using a normal Python venv:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Then start the app:

```bash
uvicorn ai_python2:app --reload
```

---

## Local Development Notes

- The project is packaged under `src/` to keep imports consistent with installed package behavior.
- MongoDB must be reachable from the environment where the app is running.
- The vector index in MongoDB should exist with the field used by the app: `embedding`.
- The app assumes a collection and index structure compatible with `$vectorSearch`.
- The checkpointer is currently `InMemorySaver()`, which means session memory is in-process and not persistent across app restarts.

---

## Important Implementation Details

### Vector search

The retrieval logic lives in `UtilFuncs.py` and does the following:

- chunks input text with `RecursiveCharacterTextSplitter`
- creates embeddings using `GoogleGenerativeAIEmbeddings`
- searches MongoDB with `$vectorSearch`
- limits results to the relevant `interviewID` and `sourceType`

```python
filter_doc = {"interviewID": ObjectId(interviewID)}
if source_type:
    filter_doc["sourceType"] = source_type
```

This allows the model to answer questions grounded in the candidate's specific resume, job description, or self-description without mixing unrelated records.

### Agent behavior

The chat agent is not a generic freeform answer generator. It is configured with a prompt that explicitly instructs it to:

- prefer retrieval tools for candidate/job details
- use the correct source-specific tool
- avoid guessing when information is missing
- use Tavily only for external/public information

The result is a retrieval-augmented interview assistant rather than a generic chatbot.

---

## Example Use Cases

- Compare candidate experience to job requirements
- Identify missing qualifications in the resume
- Summarize candidate strengths from self-description and resume
- Answer role-specific questions using stored interview context
- Cross-check job requirements against resume evidence

---

## Limitations

- Session memory is in-memory and resets when the server restarts.
- The chat flow depends on valid Google API and Tavily credentials.
- MongoDB vector indexing must be configured correctly for `$vectorSearch` to work.
- Retrieval quality depends on how clean and structured the input text is.

---

## Summary

This project is a small but practical RAG application for interview preparation and resume-to-role matching. It combines FastAPI, MongoDB vector search, and LangChain tool-calling agents to create an assistant that is anchored to a candidate's own data and the specific role they are interviewing for.

If you want to extend this project further, the most natural next steps are:

- persist session memory in a database instead of in-memory storage
- add user authentication and role-based access
- support multiple interviews per user
- add a front-end dashboard for submitting and querying interview data
- improve evaluation and observability around LLM/tool usage
