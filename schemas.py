"""API contracts for the model gateway. The UI and promptfoo tests build against these; change them only with a version bump."""
from typing import Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = "v1"


class InferenceRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="Customer question, free text")
    channel: Literal["web", "mobile", "branch", "internal"] = "web"
    purpose: str = Field("customer_faq", pattern=r"^[a-z0-9_]+$",
                         description="Which approved assistant answers, e.g. customer_faq or fd_assistant")
    max_new_tokens: int = Field(256, ge=16, le=512)


class Citation(BaseModel):
    faq_id: str = Field(description="Stable content-derived FAQ identifier")
    question: str = Field(description="The FAQ question that was retrieved")
    score: float = Field(description="Cosine similarity between the customer question and the FAQ")
    dataset_version: int = Field(description="Delta table version the FAQ was indexed from")


class ModelInfo(BaseModel):
    model_version: str = Field(description="Registered model name and version serving this request")
    base_model: str
    adapter: str
    adapter_sha256: str
    index_dataset_version: int
    embed_model: str


class DatasetRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(description="Where the data comes from, e.g. a file or system of record")
    owner: str = Field(description="Accountable data owner")
    purpose: str = Field(pattern=r"^[a-z0-9_]+$", description="Approved use of this data, e.g. customer_faq, fd_assistant")
    classification: Literal["public", "internal", "confidential", "restricted"]
    permission_basis: str = Field(description="Why the bank may use this data for this purpose")
    retention: str = Field(description="How long the data and its derivatives may be kept")
    parent_id: str | None = Field(None, description="For a team dataset: the approved dataset it selects FAQs from")
    keywords: list[str] = Field(default_factory=list, description="For a team dataset: words that select its FAQs")


class AssistantCreateRequest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$", description="Short id, e.g. fd_assistant")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(max_length=300)
    dataset_id: str = Field(description="Approved dataset whose knowledge the assistant answers from")


class AssistantReviewRequest(BaseModel):
    status: Literal["approved", "rejected"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)


class AssignRequest(BaseModel):
    assistants: list[str] = Field(description="Assistant ids this user may use, e.g. ['customer_faq']")


class RunRequest(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$", description="Run and adapter folder name, e.g. llama_v3")
    dataset_id: str
    base_model: str
    config: str = "configs/training/cuda-qlora.yaml"
    seed: int = 42


class ModelRegisterRequest(BaseModel):
    version: str = Field(pattern=r"^[a-z0-9_]+$")
    run_id: str = Field(description="Completed training run that produced the adapter")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: dict = Field(description="Evaluation scores for the release decision")
    notes: str = ""


class ModelReviewRequest(BaseModel):
    status: Literal["approved", "rejected"]


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["good", "bad"]
    comment: str = Field("", max_length=1000)


class InferenceResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] = Field(description="From retrieval strength; low answers are withheld")
    escalation_required: bool = Field(description="True when the customer should be sent to PhoneBanking or a branch")
    missing_information: str | None = Field(None, description="Why the assistant could not answer, when it could not")
    policy_flags: list[str] = Field(description="Guardrail findings, e.g. prompt_injection, unsupported_specifics")
    model: ModelInfo
    latency_ms: int
