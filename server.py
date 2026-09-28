"""FastAPI server for the HDFC FAQ assistant, with the review UI on the home page.

Run locally:  uvicorn server:app --port 7860
UI:           http://localhost:7860/
API docs:     http://localhost:7860/docs
"""
import os

import gradio as gr
from fastapi import FastAPI, Header, HTTPException

import registry
from inference import Assistant
from schemas import InferenceRequest, InferenceResponse
from ui import build_ui

app = FastAPI(title="HDFC FAQ Assistant API", version="1.0")

# Promote and rollback change which model is live, so they need this key (set ADMIN_KEY on the server)
ADMIN_KEY = os.getenv("ADMIN_KEY")

# Load the live model once when the server starts (takes ~30 seconds)
assistant = Assistant()


def check_admin(key):
    if not ADMIN_KEY or key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin key required")


def reload_live_model():
    # Load whichever version the registry now marks as live
    global assistant
    assistant = Assistant()


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_version": assistant.info.model_version,
            "adapter_sha256": assistant.info.adapter_sha256}


@app.post("/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    return assistant.answer(request.question, request.max_new_tokens)


@app.get("/v1/models")
def list_models():
    data = registry.load_registry()
    return {"live": data["live"], "previous": data["previous"], "versions": data["versions"],
            "history": data["history"]}


@app.post("/v1/models/{version}/promote")
def promote_model(version: str, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    try:
        registry.promote(version)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    reload_live_model()
    return health()


@app.post("/v1/models/rollback")
def rollback_model(x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    try:
        registry.rollback()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    reload_live_model()
    return health()


@app.get("/v1/evaluations")
def evaluations():
    # Evaluation scores are recorded in the registry for each released version
    data = registry.load_registry()
    return {name: entry["evaluation"] for name, entry in data["versions"].items()}


# The UI calls the same assistant as the API. The lambda looks up `assistant` on every call,
# so after a promote or rollback the UI uses the new model too.
app = gr.mount_gradio_app(app, build_ui(lambda question: assistant.answer(question)), path="/")
