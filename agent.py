import logging
from typing import Dict, TypedDict, List, Any
from langgraph.graph import StateGraph , END
from rag import FaqIndex,ANSWER_THRESHOLD,HIGH_CONFIDENCE,rag_messages
from guardrails import check_input,check_output,search_text
try:
  from evaluate import load_model,generate
  TOKENIZER,MODEL=load_model("meta-llama/Llama-3.2-1B-Instruct","models/qwen_v1")
except Exception:
  TOKENIZER,MODEL=None,None
logger=logging.getLogger("agent_pipeline")
try:
  faq_index=FaqIndex()
except Exception as e:
  logger.error(f"Failed to initialize dynamic RAG data index structure:{str(e)}")
  faq_index=None
class AgentState(TypedDict):
  customer_query:str
  scrubbed_qurey:str
  security_status:str
  intent_classification:str
  retrieved_conext:str
  policy_flags:List[str]
  citations:List[Dict[str,Any]]
  confidence:str
  escalation_required:bool
  final_support_response:str

## Node 1: The Front-Door Security Guard

def input_safety_guardrail_node(state:AgentState)-> Dict[str,Any]:
  raw_query=state["customer_query"].strip()

  masked_text,triggered_flags=check_input(raw_query)
  clean_search_query=search_text(masked_text)

  security_verdict = "CLEARED"
  current_flags=list(triggered_flags)

  if "prompt_injection" in triggered_flags:
    security_verdict="PROMPT_INJECTION_VIOLATION"
  elif "transaction_request" in triggered_flags:
    security_verdict="TRANSACTION_REQUEST_VIOLATION"

  return {
    "scrubbed_query":clean_search_query,
    "security_status":security_verdict,
    "policy_flags":current_flags
  }

# Node 2: The Sorter (Triage)

def domain_triage_node(state:AgentState) -> Dict[str,Any]:
  query=state["scrubbed_qurey"].lower().strip()
  off_topic_triggers=["reciepe","politics","weather","election","cricket","movie"]
  if any(trigger in query for trigger in off_topic_triggers):
    return {"intent_classification":"out_of_scope"}

  domain_keywords={
    "cards":["card","pin","cvv","expiry","delivery","credit","debit","blocked","stolen","lost"],
    "account":["account","balance","open","close","saving","checking","kyc","profile","passbook"],
    "deposit":["deposit","fd","rd","fixed deposit","recurring","tenure","maturity","interest","payout"],
    "loan":["loan","emi","mortgage","borrow","principal","car loan","home loan","personal loan","moratirium"],
    "policies":["policy","terms","rules","regulations","fee","charge","legal","clause","gst"]
               
  }

  domain_scores={}
  for domain,keywords in domain_keywords.items():
    match_count=sum(1 for word in keywords if word in query)
    domain_scores[domain]=match_count
  highest_score=max(domain_scores.values())

  if highest_score==0:
    detected_intent= "general_banking_query"
  else:
    winning_domains=[domain for domain,score in domain_scores.iteams()if score==highest_score]
    primary_match=winning_domains[0]

    intent_mapping={
      "cards":"cards_query",
      "account":"account_management_query",
      "deposit":"deposits_query",
      "loan":"loans_query",
      "policies":"bank_policy_query"
    }
    detected_intent=intent_mapping[primary_match]
  return {"intent_classification":detected_intent}

# Node 3: The Context Retrieval & Extraction Core (RAG)

def faq_retrieval_node(state: AgentState) -> Dict[str, Any]:
    intent = state["intent_classification"]
    
    # Bypasses database retrieval completely if triage marked it out of scope
    if intent == "out_of_scope" or faq_index is None:
        return {"retrieved_context": "", "citations": [], "confidence": "low", "escalation_required": True}
        
    try:
        # Pulls the closest matching database records from your Parquet snapshots
        search_results_list = faq_index.search([state["scrubbed_query"]])
        if not search_results_list or len(search_results_list) == 0:
            return {"retrieved_context": "", "citations": [], "confidence": "low", "escalation_required": True}
            
        raw_retrieved = search_results_list[0]
        
        valid_chunks = []
        valid_citations = []
        highest_score = 0.0
        
        # Filter matches and build metadata dictionaries safely
        for faq_row, score_val in raw_retrieved:
            try:
                score = float(score_val)
            except (ValueError, TypeError):
                score = 0.0
                
            # Keep the answer only if it passes  absolute quality threshold
            if score >= ANSWER_THRESHOLD:
                valid_chunks.append(faq_row.get("Target_Banking_Response", ""))
                
                try:
                    ds_version = int(faq_index.meta.get("dataset_version", 7))
                except Exception:
                    ds_version = 7
                    
                valid_citations.append({
                    "faq_id": faq_row.get("faq_id", "unknown"),
                    "question": faq_row.get("User_Query", ""),
                    "score": score,
                    "dataset_version": ds_version
                })
                if score > highest_score:
                    highest_score = score
                    
        # Applying strict contract confidence tiers to control human escalation tags
        if highest_score >= HIGH_CONFIDENCE:
            confidence_tier = "high"
            escalation = False
        elif highest_score >= ANSWER_THRESHOLD:
            confidence_tier = "medium"
            escalation = False
        else:
            confidence_tier = "low"
            escalation = True
            
        combined_context = "\n\n".join(valid_chunks).strip()
        return {
            "retrieved_context": combined_context,
            "citations": valid_citations,
            "confidence": confidence_tier,
            "escalation_required": escalation
        }
    except Exception as e:
        logger.error(f"FAQ Index matrix retrieval fault: {str(e)}")
        return {"retrieved_context": "", "citations": [], "confidence": "low", "escalation_required": True}

#Node 4: The Response Generator & Editor
def response_generation_node(state: AgentState) -> Dict[str, Any]:
    security = state.get("security_status", "CLEARED")
    intent = state.get("intent_classification", "general_banking_query")
    current_flags = list(state.get("policy_flags", []))
    citations = list(state.get("citations", []))
    context = state.get("retrieved_context", "")
    query = state["scrubbed_query"]
    
    escalation = state.get("escalation_required", False)
    confidence = state.get("confidence", "high")

    # 1. Input Guardrail Enforcements (Controlled Refusals)
    if security == "PROMPT_INJECTION_VIOLATION":
        if "prompt_injection" not in current_flags:
            current_flags.append("prompt_injection")
        return {
            "final_support_response": "I can only help with questions about HDFC Bank.",
            "policy_flags": current_flags, "escalation_required": False, "confidence": "high", "citations": citations
        }
        
    if security == "TRANSACTION_REQUEST_VIOLATION":
        if "transaction_request" not in current_flags:
            current_flags.append("transaction_request")
        return {
            "final_support_response": "I can't carry out transactions directly.",
            "policy_flags": current_flags, "escalation_required": True, "confidence": "high", "citations": citations
        }
        
    if intent == "out_of_scope":
        if "out_of_scope" not in current_flags:
            current_flags.append("out_of_scope")
        return {
            "final_support_response": "I am an authorized HDFC banking assistant and can only resolve financial service queries.",
            "policy_flags": current_flags, "escalation_required": True, "confidence": "low", "citations": citations
        }

    # 2. Rejection Fallback for Low Relevance / Missing Data
    if escalation or not context or confidence == "low":
        return {
            "final_support_response": "I do not have that information available right now. Please contact HDFC Bank PhoneBanking or visit the nearest branch.",
            "policy_flags": current_flags, "escalation_required": True, "confidence": "low", "citations": citations
        }

    # 3. Execution Generation Core (Natively loops context strings)
    retrieved_data_list = []
    for c in citations:
        mock_row = {
            "faq_id": c["faq_id"], 
            "User_Query": c["question"], 
            "Target_Banking_Response": context
        }
        retrieved_data_list.append((mock_row, c["score"]))

    if MODEL and TOKENIZER:
        try:
            messages_payload = rag_messages(query, retrieved_data_list)
            generation_outputs = generate(MODEL, TOKENIZER, [messages_payload], max_new_tokens=256, batch_size=1)
            raw_output_text = generation_outputs if isinstance(generation_outputs, list) else generation_outputs
        except Exception as e:
            logger.error(f"Inference pipeline execution fault: {str(e)}")
            raw_output_text = f"Based on HDFC records: {context}"
    else:
        raw_output_text = f"Based on official reference data, the protocol details state that: {context}"

    # 4. Final Output Compliance Evaluation Check
    safe_answer, output_flags, _ = check_output(raw_output_text, context)
    
    for flag in output_flags:
        if flag not in current_flags:
            current_flags.append(flag)
            
    # Apply unique orchestration behavior modifications per intent group
    if intent == "cards_query":
        escalation = True

    return {
        "final_support_response": safe_answer,
        "policy_flags": current_flags,
        "escalation_required": escalation,
        "confidence": confidence,
        "citations": citations
    }
# The Routing Switch function
def security_routing_router(state: AgentState) -> str:
    if state["security_status"] != "CLEARED":
        return "route_to_refusal"
    return "route_to_triage"

# Assemblling the final LangGraph layout loop
workflow = StateGraph(AgentState)

workflow.add_node("safety_gate", input_safety_guardrail_node)
workflow.add_node("triage_intent", domain_triage_node)
workflow.add_node("extract_facts", faq_retrieval_node)
workflow.add_node("generate_output", response_generation_node)

workflow.set_entry_point("safety_gate")

workflow.add_conditional_edges(
    "safety_gate",
    security_routing_router,
    {
        "route_to_refusal": "generate_output",
        "route_to_triage": "triage_intent"
    }
)

workflow.add_edge("triage_intent", "extract_facts")
workflow.add_edge("extract_facts", "generate_output")
workflow.add_edge("generate_output", END)

compiled_workflow = workflow.compile()

