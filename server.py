"""FastAPI server for the HDFC AI Platform: login, control-plane APIs, the model gateway and the web UI.

Run locally:  uvicorn server:app --port 7860
Web UI:       http://localhost:7860/
API docs:     http://localhost:7860/docs
"""
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

import auth
import control_plane
import registry
from inference import Assistant
from rag import FaqIndex
from schemas import (AssignRequest, AssistantCreateRequest, AssistantReviewRequest, DatasetRegisterRequest,
                     FeedbackRequest, InferenceRequest, InferenceResponse, LoginRequest, ModelRegisterRequest,
                     ModelReviewRequest, RunRequest)

app = FastAPI(title="HDFC AI Platform API", version="1.0")

# Applications (not people) call the gateway with this key. It comes from the environment, never from code.
APP_KEY = os.getenv("APP_KEY")
# Optional token Prometheus sends when scraping /metrics
METRICS_TOKEN = os.getenv("METRICS_TOKEN")

# Load the live model once when the server starts (takes ~30 seconds)
assistant = Assistant()


def reload_live_model():
    # Load whichever version the registry now marks as live
    global assistant
    assistant = Assistant()


team_indexes = {}  # index folder -> loaded FaqIndex, so each team index is read from disk once


def index_for(assistant_id):
    """The knowledge index an assistant answers from (None means the full FAQ index)."""
    index_dir = control_plane.get_assistant(assistant_id)["index_dir"]
    if index_dir == "models/rag_index":
        return None
    if index_dir not in team_indexes:
        team_indexes[index_dir] = FaqIndex(Path(index_dir), embedder=assistant.index.embedder)
    return team_indexes[index_dir]


# ---------- Helpers: who is calling, and are they allowed? ----------

def current_user(authorization):
    """Read 'Authorization: Bearer <token>' and return the logged-in user."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    user = auth.user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in")
    return user


def require(authorization, permission):
    user = current_user(authorization)
    if not auth.can(user, permission):
        control_plane.audit(user["username"], permission, "-", "denied", "Role not allowed")
        raise HTTPException(status_code=403, detail=f"Your role ({user['role']}) cannot {permission.replace('_', ' ')}")
    return user


def act(user, action, resource, function, *args):
    """Run a control-plane action, record it in the audit log, and turn errors into HTTP errors."""
    try:
        result = function(*args)
    except KeyError as error:
        control_plane.audit(user["username"], action, resource, "failed", str(error).strip("'\""))
        raise HTTPException(status_code=404, detail=str(error).strip("'\""))
    except ValueError as error:
        control_plane.audit(user["username"], action, resource, "failed", str(error))
        raise HTTPException(status_code=400, detail=str(error))
    control_plane.audit(user["username"], action, resource, "success")
    return result


def read(function, *args):
    """Run a read-only lookup; unknown ids become 404."""
    try:
        return function(*args)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'\""))


# ---------- Login ----------

@app.post("/v1/auth/login")
def login(request: LoginRequest):
    try:
        token, user = auth.login(request.username, request.password)
    except auth.TooManyAttempts as error:
        control_plane.audit(request.username, "login", "-", "denied", str(error))
        raise HTTPException(status_code=429, detail=str(error))
    except ValueError as error:
        control_plane.audit(request.username, "login", "-", "failed", str(error))
        raise HTTPException(status_code=401, detail=str(error))
    control_plane.audit(user["username"], "login", "-", "success")
    return {"token": token, "user": user}


@app.post("/v1/auth/logout")
def logout(authorization: str = Header(None)):
    user = current_user(authorization)
    auth.logout((authorization or "").removeprefix("Bearer ").strip())
    control_plane.audit(user["username"], "logout", "-", "success")
    return {"status": "logged out"}


@app.get("/v1/auth/me")
def me(authorization: str = Header(None)):
    return current_user(authorization)


# ---------- Health and assistants ----------

@app.get("/v1/health")
def health():
    return {"status": "ok", "model_version": assistant.info.model_version,
            "adapter_sha256": assistant.info.adapter_sha256}


@app.get("/v1/assistants")
def list_assistants(authorization: str = Header(None)):
    """Platform users see every assistant; employees see only approved assistants assigned to them."""
    user = current_user(authorization)
    result = []
    for key, value in control_plane.list_assistants().items():
        visible = auth.can(user, "view_platform") or (value["status"] == "approved" and auth.can_use_assistant(user, key))
        if visible:
            result.append({"id": key, **value, "model_version": assistant.info.model_version})
    return result


@app.post("/v1/assistants")
def create_assistant(request: AssistantCreateRequest, authorization: str = Header(None)):
    user = require(authorization, "create_assistant")
    return act(user, "create_assistant", request.id, control_plane.create_assistant, request.id, request.name,
               request.description, request.dataset_id, user["username"])


@app.post("/v1/assistants/{assistant_id}/review")
def review_assistant(assistant_id: str, request: AssistantReviewRequest, authorization: str = Header(None)):
    user = require(authorization, "review_assistant")
    return act(user, f"{request.status}_assistant", assistant_id, control_plane.review_assistant, assistant_id,
               request.status, user["username"])


# ---------- Users (admin) ----------

@app.get("/v1/users")
def list_users(authorization: str = Header(None)):
    require(authorization, "manage_users")
    return [auth.public(user) for user in auth.load_users()]


@app.post("/v1/users/{username}/assistants")
def assign_assistants(username: str, request: AssignRequest, authorization: str = Header(None)):
    admin = require(authorization, "manage_users")

    def assign():
        approved = {k for k, v in control_plane.list_assistants().items() if v["status"] == "approved"}
        not_allowed = set(request.assistants) - approved
        if not_allowed:
            raise ValueError(f"Only approved assistants can be assigned: {sorted(not_allowed)}")
        users = auth.load_users()
        for user in users:
            if user["username"] == username:
                user["assistants"] = request.assistants
                auth.save_users(users)
                return auth.public(user)
        raise KeyError(f"Unknown user: {username}")

    return act(admin, "assign_assistants", username, assign)


# ---------- Datasets ----------

@app.get("/v1/datasets")
def list_datasets(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return control_plane.list_datasets()


@app.get("/v1/datasets/{dataset_id}")
def get_dataset(dataset_id: str, authorization: str = Header(None)):
    require(authorization, "view_platform")
    return read(control_plane.get_dataset, dataset_id)


@app.post("/v1/datasets")
def register_dataset(request: DatasetRegisterRequest, authorization: str = Header(None)):
    user = require(authorization, "register_dataset")
    return act(user, "register_dataset", request.name, control_plane.register_dataset, request.name, request.source,
               request.owner, request.purpose, request.classification, request.permission_basis, request.retention,
               request.parent_id, request.keywords)


@app.post("/v1/datasets/{dataset_id}/prepare")
def prepare_dataset(dataset_id: str, authorization: str = Header(None)):
    user = require(authorization, "prepare_dataset")
    return act(user, "prepare_dataset", dataset_id, control_plane.request_preparation, dataset_id)


@app.post("/v1/datasets/{dataset_id}/approve")
def approve_dataset(dataset_id: str, authorization: str = Header(None)):
    user = require(authorization, "approve_dataset")
    return act(user, "approve_dataset", dataset_id, control_plane.approve_dataset, dataset_id, user["username"])


# ---------- Training runs ----------

@app.get("/v1/runs")
def list_runs(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return control_plane.list_runs()


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, authorization: str = Header(None)):
    require(authorization, "view_platform")
    return read(control_plane.get_run, run_id)


@app.get("/v1/base-models")
def base_models(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return {"approved_base_models": control_plane.APPROVED_BASE_MODELS,
            "configs": sorted(str(p).replace("\\", "/") for p in Path("configs/training").glob("*.yaml"))}


@app.post("/v1/runs")
def request_run(request: RunRequest, authorization: str = Header(None)):
    user = require(authorization, "request_run")
    return act(user, "request_run", request.name, control_plane.request_run, request.name, request.dataset_id,
               request.base_model, request.config, request.seed)


# ---------- Evaluations and monitoring ----------

@app.get("/v1/evaluations")
def evaluations(authorization: str = Header(None)):
    require(authorization, "view_platform")
    data = registry.load_registry()
    return {name: entry["evaluation"] for name, entry in data["versions"].items()}


@app.get("/v1/monitoring")
def monitoring(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return control_plane.monitoring_summary()


@app.get("/metrics", include_in_schema=False)
def metrics(authorization: str = Header(None)):
    """Prometheus scrape endpoint. If METRICS_TOKEN is set, Prometheus must send it as a bearer token."""
    if METRICS_TOKEN and (authorization or "") != f"Bearer {METRICS_TOKEN}":
        raise HTTPException(status_code=401, detail="Metrics token required")
    return PlainTextResponse(control_plane.prometheus_metrics(assistant.info.model_version))


@app.get("/v1/audit")
def audit_log(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return control_plane.audit_log()


# ---------- Model registry and deployment ----------

@app.get("/v1/models")
def list_models(authorization: str = Header(None)):
    require(authorization, "view_platform")
    return registry.load_registry()


@app.post("/v1/models/register")
def register_model(request: ModelRegisterRequest, authorization: str = Header(None)):
    user = require(authorization, "register_model")

    def register():
        run = control_plane.get_run(request.run_id)
        if run["status"] != "completed":
            raise ValueError(f"Run {request.run_id} has not completed")
        return registry.register_model(request.version, request.run_id, request.adapter_sha256,
                                       request.evaluation, request.notes)

    return act(user, "register_model", request.version, register)


@app.post("/v1/models/{version}/review")
def review_model(version: str, request: ModelReviewRequest, authorization: str = Header(None)):
    user = require(authorization, "review_model")
    return act(user, f"{request.status}_model", version, registry.set_status, version, request.status,
               user["username"])


@app.post("/v1/models/{version}/promote")
def promote_model(version: str, authorization: str = Header(None)):
    user = require(authorization, "promote")
    act(user, "promote", version, registry.promote, version, registry.REGISTRY_PATH, user["username"])
    reload_live_model()
    return health()


@app.post("/v1/deployments/{deployment_id}/rollback")
def rollback_deployment(deployment_id: str, authorization: str = Header(None)):
    user = require(authorization, "rollback")
    if deployment_id != "production":
        raise HTTPException(status_code=404, detail="Only the 'production' deployment exists")
    act(user, "rollback", deployment_id, registry.rollback, registry.REGISTRY_PATH, user["username"])
    reload_live_model()
    return health()


# ---------- Model gateway and feedback ----------

def gateway_caller(authorization, x_api_key, assistant_id):
    """Applications use the app key; people use their login and must be assigned the assistant."""
    try:
        status = control_plane.get_assistant(assistant_id)["status"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown assistant: {assistant_id}")
    if status != "approved":
        raise HTTPException(status_code=403, detail=f"Assistant {assistant_id} is not approved yet")
    if APP_KEY and x_api_key == APP_KEY:
        caller = "application"
    else:
        user = current_user(authorization)
        if not auth.can_use_assistant(user, assistant_id):
            control_plane.audit(user["username"], "use_assistant", assistant_id, "denied", "Assistant not assigned")
            raise HTTPException(status_code=403, detail="This assistant is not assigned to you")
        caller = user["username"]
    try:
        auth.allow_request(caller)
    except auth.TooManyAttempts as error:
        control_plane.audit(caller, "use_assistant", assistant_id, "denied", "Rate limit reached")
        raise HTTPException(status_code=429, detail=str(error))
    return caller


@app.post("/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest, authorization: str = Header(None), x_api_key: str = Header(None)):
    caller = gateway_caller(authorization, x_api_key, request.purpose)
    started = time.perf_counter()
    try:
        response = assistant.answer(request.question, request.max_new_tokens, index=index_for(request.purpose))
    except Exception as error:
        # Record the failure for the success-rate SLO, then return a controlled error instead of a stack trace
        trace_id = uuid.uuid4().hex
        control_plane.log_failed_request(trace_id, caller, request.purpose, assistant.info.model_version,
                                         round((time.perf_counter() - started) * 1000), error)
        raise HTTPException(status_code=503, detail=f"The assistant could not answer right now (trace {trace_id}). "
                                                    "Please try again or contact PhoneBanking.")
    control_plane.log_request(response, caller, request.purpose)
    return response


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest, authorization: str = Header(None), x_api_key: str = Header(None)):
    # Feedback is allowed from whoever may use the assistant that gave the answer
    assistant_id = read(control_plane.find_request, request.trace_id)["assistant"]
    caller = gateway_caller(authorization, x_api_key, assistant_id)
    return read(control_plane.add_feedback, request.trace_id, request.rating, request.comment, caller)


# ---------- Web UI ----------

@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path("frontend/index.html"))
