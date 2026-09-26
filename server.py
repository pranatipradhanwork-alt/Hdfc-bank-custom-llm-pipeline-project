import time

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import compiled_workflow

app = FastAPI(
    title="HDFC Customer Support LLM API Gateway",
    description="REST endpoint for our LangGraph pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserInferenceRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    channel: str = Field(default="web")
    purpose: str = Field(default="customer_faq")
    max_new_tokens: int = Field(default=256, ge=16, le=512)


@app.post("/v1/inference", status_code=status.HTTP_200_OK)
async def platform_inference_endpoint(payload: UserInferenceRequest):
    """Run a customer question through the workflow and return its response."""
    start_time = time.time()

    graph_inputs = {
        "customer_query": payload.question,
        "scrubbed_query": "",
        "security_status": "",
        "intent_classification": "",
        "retrieved_context": "",
        "policy_flags": [],
        "citations": [],
        "confidence": "",
        "escalation_required": False,
        "final_support_response": "",
    }

    try:
        final_state = compiled_workflow.invoke(graph_inputs)
        latency = int((time.time() - start_time) * 1000)

        return {
            "trace_id": "example-trace-id",
            "answer": final_state.get("final_support_response"),
            "citations": final_state.get("citations", []),
            "confidence": final_state.get("confidence", "high"),
            "escalation_required": final_state.get("escalation_required", False),
            "missing_information": None,
            "policy_flags": final_state.get("policy_flags", []),
            "model": {
                "model_version": "hdfc-faq-assistant/2",
                "base_model": "meta-llama/Llama-3.2-1B-Instruct",
                "adapter": "models/llama_v2",
                "adapter_sha256": "stable_pipeline_hash_metric",
                "index_dataset_version": 7,
                "embed_model": "BAAI/bge-small-en-v1.5",
            },
            "latency_ms": latency,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph pipeline execution failed: {str(e)}",
        ) from e


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0", port=8000, reload=True)