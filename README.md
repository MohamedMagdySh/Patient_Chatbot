# 🏥 MediBot — Medical Triage Chatbot

A medical triage chatbot for patients, powered by Google Gemini AI and RAG, with support for Arabic, English, and mixed input — includes Egyptian drug prices and Ministry of Health protocols.

## Features

- Accepts symptoms in Arabic, English, or mixed
- Retrieves relevant context from Egyptian medical knowledge base
- Suggests OTC medications with local trade names and prices
- Follows Egyptian MOH treatment protocols
- Streaming responses support
- Multi-session conversation management

## Tech Stack

- **FastAPI** — REST API backend
- **Google Gemini 2.5 Flash** — AI model
- **FAISS** — Vector similarity search (RAG)
- **Sentence Transformers** — Text embeddings

-----------------------------------------------------------------------
## Project Structure
├── main.py         # FastAPI app and endpoints
├── config.py       # Settings and AI system prompt
├── rag.py          # FAISS index and retrieval
├── chat.py         # Gemini sessions and chat logic
├── requirements.txt
└── researches/     # Egyptian medical knowledge base

-------------------------------------------------------------------------
## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status check |
| POST | `/chat` | Send message, get full reply |
| POST | `/chat/stream` | Send message, get streamed reply |
| POST | `/reset` | Clear conversation history |
| GET | `/stats/{session_id}` | Token usage and message count |

## Setup

1. Clone the repository
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Create `.env` file
```env
GEMINI_API_KEY=your_key_here
RESEARCH_FOLDER=researches
ALLOWED_ORIGINS=*
```
4. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Disclaimer

This chatbot provides preliminary medical assessments only and does not replace consultation with a licensed physician.
