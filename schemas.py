"""API contracts for the model gateway. The UI and promptfoo tests build against these; change them only with a version bump."""
from typing import Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = "v1"


class InferenceRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="Customer question, free text")
    channel: Literal["web", "mobile", "branch", "internal"] = "web"
    purpose: Literal["customer_faq"] = Field("customer_faq", description="Only approved purpose in this release")
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
