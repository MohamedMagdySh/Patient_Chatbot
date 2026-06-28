# Manages Gemini AI sessions, conversation history, and sending/streaming messages

import time
import random
import logging

from google import genai
from google.genai import types
from fastapi import HTTPException

from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from rag import retrieve_context

logger   = logging.getLogger(__name__)
client   = genai.Client(api_key=GEMINI_API_KEY)
sessions = {}


def get_or_create_session(session_id: str) -> dict:
    # Returns existing session or creates a new one
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "tokens": 0}
    return sessions[session_id]


def reset_session(session_id: str):
    # Clears conversation history for a session
    sessions[session_id] = {"history": [], "tokens": 0}


def _build_prompt(user_message: str) -> str:
    # Attaches relevant RAG context to the user message if available
    context = retrieve_context(user_message)
    if context:
        return f"[Relevant Research]\n{context}\n\n[Patient Message]\n{user_message}"
    return f"[Patient Message]\n{user_message}"


def _do_chat(session: dict, user_message: str) -> str:
    # Sends message to Gemini and appends both turns to session history
    prompt = _build_prompt(user_message)
    session["history"].append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=session["history"],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.7),
    )

    reply = response.text
    session["history"].append(types.Content(role="model", parts=[types.Part(text=reply)]))

    try:
        session["tokens"] += response.usage_metadata.total_token_count
    except Exception:
        pass

    return reply


def chat_with_retry(session: dict, user_message: str, max_retries: int = 4) -> str:
    # Retries the chat request with exponential backoff on errors
    for attempt in range(max_retries):
        try:
            return _do_chat(session, user_message)

        except Exception as e:
            error_str = str(e).lower()

            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit. Retry in {wait:.1f}s")
                time.sleep(wait)

            elif "400" in error_str or "invalid" in error_str:
                raise HTTPException(status_code=400, detail="Invalid request.")

            elif "504" in error_str or "timeout" in error_str or "deadline" in error_str:
                raise HTTPException(status_code=504, detail="Request timed out.")

            elif "503" in error_str or "unavailable" in error_str:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Service unavailable. Retry in {wait:.1f}s")
                time.sleep(wait)

            else:
                logger.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error.")

    raise HTTPException(status_code=503, detail="Service unavailable.")


def stream_chat(session: dict, user_message: str):
    # Streams the Gemini response chunk by chunk in SSE format
    prompt = _build_prompt(user_message)
    session["history"].append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    full_reply = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=session["history"],
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.7),
        ):
            if chunk.text:
                full_reply += chunk.text
                yield f"data: {chunk.text.replace(chr(10), chr(92)+'n')}\n\n"

        session["history"].append(types.Content(role="model", parts=[types.Part(text=full_reply)]))
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"
