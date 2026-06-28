# FastAPI app — defines all API endpoints and starts the server

import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import ALLOWED_ORIGINS, GEMINI_MODEL
from rag import init_rag
from chat import get_or_create_session, reset_session, sessions, chat_with_retry, stream_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="MediBot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    # Load or build RAG index when server starts
    init_rag()


# Request / Response schemas
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tokens_used: int

class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    # Returns server status
    return {"status": "ok", "model": GEMINI_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Receives a patient message and returns the full MediBot reply
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    session    = get_or_create_session(session_id)
    reply      = chat_with_retry(session, req.message)

    return ChatResponse(reply=reply, session_id=session_id, tokens_used=session["tokens"])


@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    # Same as /chat but streams the reply token by token via SSE
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    session    = get_or_create_session(session_id)

    def generate():
        yield f"data: [SESSION_ID]{session_id}[/SESSION_ID]\n\n"
        yield from stream_chat(session, req.message)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    # Clears conversation history for a given session
    reset_session(req.session_id)
    return {"session_id": req.session_id, "message": "Session reset successfully."}


@app.get("/stats/{session_id}")
def stats_endpoint(session_id: str):
    # Returns token usage and message count for a session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = sessions[session_id]
    return {
        "session_id":     session_id,
        "tokens_used":    session["tokens"],
        "history_length": len(session["history"]),
    }
