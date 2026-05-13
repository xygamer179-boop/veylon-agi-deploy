#!/usr/bin/env python3
"""
Veylon AGI v5 — Production FastAPI Server
Python 3.9–3.11 compatible | Render / HuggingFace ready
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn, os, sys, traceback, time, math, asyncio, re
from typing import Optional

# ── Import Veylon (adjust if your file has a different name) ──
import veylon_agi_v5 as veylon

# ── Pydantic models ───────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str
    mode: str = "agent"
    image_url: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    partial: bool = False
    error: str = ""

class SplitResponse(BaseModel):
    thinking: str
    output: str
    partial: bool = False
    error: str = ""

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: float
    intent: str = "chat"

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Veylon AGI API", version=veyon.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    if veylon._model is None:
        veylon._load_model()
    print("✅ Veylon 80M API ready")

# ── Response cleaner (remove technical footers) ───────────────
def clean_response(text: str) -> str:
    """Strip [ intent=... ] and URP lines from output."""
    text = re.sub(r"\n?\n\[ intent=[^\]]+ \]", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\[URP\].*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── Safe wrapper — never crashes ──────────────────────────────
def safe_process(text, mode, image_url=None):
    partial = False
    error_msg = ""
    try:
        full = veylon.process_input(
            user_input=text,
            mode_override=mode,
            image_url=image_url,
        )
        return clean_response(full), partial, error_msg
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        partial = True
        try:
            sparse = veylon._encode(text)
            intent, _ = veylon.detect_intent(text, veylon._model, sparse)
            return (
                f"Partial response (recovery mode)\n"
                f"Intent: {intent}\n"
                f"Error: {error_msg}",
                True,
                error_msg,
            )
        except:
            return (
                "Veylon temporarily unavailable. Please try again.",
                True,
                error_msg,
            )

# ── Thinking / Output splitter ────────────────────────────────
def split_thinking_output(text: str):
    """Separate reasoning steps from the final answer."""
    # Try splitting on the feature‑list marker
    parts = text.split("\n★", 1)
    thinking = parts[0].strip() if len(parts) > 0 else ""
    output   = ("★" + parts[1]).strip() if len(parts) > 1 else thinking

    # Fallback: split at last double newline
    if thinking == output:
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 3:
            output   = paragraphs[-1].strip()
            thinking = "\n\n".join(paragraphs[:-1]).strip()
        else:
            output   = text
            thinking = ""
    return thinking, output

# ── Endpoints ─────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    resp_text, partial, err = safe_process(req.text, req.mode, req.image_url)
    return ChatResponse(response=resp_text, partial=partial, error=err)

@app.post("/chat/split", response_model=SplitResponse)
def chat_split(req: ChatRequest):
    resp_text, partial, err = safe_process(req.text, req.mode, req.image_url)
    thinking, output = split_thinking_output(resp_text)
    return SplitResponse(thinking=thinking, output=output, partial=partial, error=err)

@app.get("/health")
def health():
    return {"status": "ok", "version": veylon.VERSION}

# ── Streaming (word‑by‑word, ChatGPT‑style) ──────────────────
async def stream_text(text: str):
    for word in text.split():
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.03)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    resp_text, _, _ = safe_process(req.text, req.mode, req.image_url)
    return StreamingResponse(stream_text(resp_text), media_type="text/event-stream")

# ── RLHF feedback ─────────────────────────────────────────────
@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    try:
        if veylon._trainer is None:
            return {"status": "RLHF unavailable"}
        sparse = veylon._encode(fb.query)
        idx = veylon.INTENT_LABELS.index(fb.intent) if fb.intent in veylon.INTENT_LABELS else veylon.INTENT_LABELS.index("chat")
        reward = max(-1.0, min(1.0, float(fb.rating)))
        with veylon._model_lock:
            loss = veylon._trainer.train_one(sparse, idx, reward=reward)
        veylon._schedule_save()
        return {"status": "applied", "reward": round(reward, 4), "loss": round(loss, 4) if not math.isnan(loss) else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
