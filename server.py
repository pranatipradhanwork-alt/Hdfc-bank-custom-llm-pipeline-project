"""FastAPI server for the HDFC FAQ assistant: control-plane APIs, the model gateway and the review UI.

Run locally:  uvicorn server:app --port 7860
UI:           http://localhost:7860/
API docs:     http://localhost:7860/docs
"""
import os

import gradio as gr
from fastapi import FastAPI, Header, HTTPException

import control_plane
import registry
from inference import Assistant
from schemas import (DatasetApproveRequest, DatasetRegisterRequest, FeedbackRequest, InferenceRequest,
                     InferenceResponse, ModelRegisterRequest, ModelReviewRequest, RunRequest)
from ui import build_ui

app = FastAPI(title="HDFC Custom LLM Pipeline API", version="1.0")

# Secrets come from the environment, never from code.
# ADMIN_KEY protects governance actions; APP_KEY protects the model gateway for applications.
ADMIN_KEY = os.getenv("ADMIN_KEY")
APP_KEY = os.getenv("APP_KEY")

# Load the live model once when the server starts (takes ~30 seconds)
assistant = Assistant()


def check_admin(key):
    if not ADMIN_KEY or key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin key required")


def check_app(key):
    # If no APP_KEY is set (local development), the gateway is open
    if APP_KEY and key != APP_KEY:
        raise HTTPException(status_code=401, detail="Application key required")


def run_action(action, *args):
    """Call a control-plane function and turn its errors into HTTP errors."""
    try:
        return action(*args)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'\""))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


def answer_and_log(question, max_new_tokens=256):
    response = assistant.answer(question, max_new_tokens)
    control_plane.log_request(response)
    return response


def reload_live_model():
    # Load whichever version the registry now marks as live
    global assistant
    assistant = Assistant()


# ---------- Health ----------

@app.get("/v1/health")
def health():
    return {"status": "ok", "model_version": assistant.info.model_version,
            "adapter_sha256": assistant.info.adapter_sha256}


# ---------- Datasets ----------

@app.get("/v1/datasets")
def list_datasets():
    return control_plane.list_datasets()


@app.get("/v1/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    return run_action(control_plane.get_dataset, dataset_id)


@app.post("/v1/datasets")
def register_dataset(request: DatasetRegisterRequest, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    return run_action(control_plane.register_dataset, request.name, request.source, request.owner, request.purpose,
                      request.classification, request.permission_basis, request.retention)


@app.post("/v1/datasets/{dataset_id}/prepare")
def prepare_dataset(dataset_id: str, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    return run_action(control_plane.request_preparation, dataset_id)


@app.post("/v1/datasets/{dataset_id}/approve")
def approve_dataset(dataset_id: str, request: DatasetApproveRequest, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    return run_action(control_plane.approve_dataset, dataset_id, request.approved_by)


# ---------- Training runs ----------

@app.get("/v1/runs")
def list_runs():
    return control_plane.list_runs()


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str):
    return run_action(control_plane.get_run, run_id)


@app.post("/v1/runs")
def request_run(request: RunRequest, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    return run_action(control_plane.request_run, request.name, request.dataset_id, request.base_model,
                      request.config, request.seed)


# ---------- Evaluations ----------

@app.get("/v1/evaluations")
def evaluations():
    # Evaluation scores are recorded in the registry for each released version
    data = registry.load_registry()
    return {name: entry["evaluation"] for name, entry in data["versions"].items()}


# ---------- Model registry ----------

@app.get("/v1/models")
def list_models():
    return registry.load_registry()


@app.post("/v1/models/register")
def register_model(request: ModelRegisterRequest, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    run = run_action(control_plane.get_run, request.run_id)
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Run {request.run_id} has not completed")
    return run_action(registry.register_model, request.version, request.run_id, request.adapter_sha256,
                      request.evaluation, request.notes)


@app.post("/v1/models/{version}/review")
def review_model(version: str, request: ModelReviewRequest, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    return run_action(registry.set_status, version, request.status, request.reviewer)


@app.post("/v1/models/{version}/promote")
def promote_model(version: str, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    run_action(registry.promote, version)
    reload_live_model()
    return health()


@app.post("/v1/deployments/{deployment_id}/rollback")
def rollback_deployment(deployment_id: str, x_admin_key: str = Header(None)):
    check_admin(x_admin_key)
    if deployment_id != "production":
        raise HTTPException(status_code=404, detail="Only the 'production' deployment exists")
    run_action(registry.rollback)
    reload_live_model()
    return health()


# ---------- Model gateway and feedback ----------

@app.post("/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest, x_api_key: str = Header(None)):
    check_app(x_api_key)
    return answer_and_log(request.question, request.max_new_tokens)


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest, x_api_key: str = Header(None)):
    check_app(x_api_key)
    return run_action(control_plane.add_feedback, request.trace_id, request.rating, request.comment, request.reviewer)


# The UI uses the same functions as the API. The lambdas look up `assistant` on every call,
# so after a promote or rollback the UI uses the new model too.
ui = build_ui(answer_question=lambda question: answer_and_log(question),
              current_model=lambda: assistant.info,
              reload_model=reload_live_model,
              admin_key=ADMIN_KEY)
app = gr.mount_gradio_app(app, ui, path="/")
