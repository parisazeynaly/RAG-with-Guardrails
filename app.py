from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from src.retriever import search
from src.guardrails import Guardrails
from src.llm_client import EchoLLM, LLMClient

app = FastAPI(title="RAG with Guardrails API")
guard = Guardrails()
llm: LLMClient = EchoLLM()  # Replace with your real implementation

class AskRequest(BaseModel):
    query: str
    top_k: int = 4

class AskResponse(BaseModel):
    decision: str
    answer: str
    contexts: List[Dict[str, Any]]
    safety_log: Dict[str, Any]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Guard the incoming user query (input guard)
    safety = guard.check(req.query)
    if safety["decision"] == "block":
        return AskResponse(decision="block", answer=guard.safe_respond(req.query), contexts=[], safety_log=safety["log"])

    # Retrieve
    ctxs = search(req.query, k=req.top_k)
    ctx_text = "\n\n".join([c[0] for c in ctxs])

    # Build RAG prompt
    prompt = f"""You are a helpful assistant. Use the provided context to answer.
Context:
{ctx_text}

User question: {req.query}
Answer briefly and cite the most relevant sources by filename.
"""

    # Call LLM
    raw_answer = llm.generate(prompt)

    # Output guard (optional)
    out_safety = guard.check(raw_answer)
    final_decision = out_safety["decision"]
    if final_decision == "block":
        final_answer = guard.safe_respond(req.query)
    elif final_decision == "filter":
        # naive filter: redact toxic terms (for demo)
        final_answer = raw_answer.replace("idiot", "[redacted]").replace("stupid", "[redacted]")
    else:
        final_answer = raw_answer

    return AskResponse(
        decision=final_decision,
        answer=final_answer,
        contexts=[{"text": t, "meta": m, "score": s} for t, m, s in ctxs],
        safety_log={"input": safety["log"], "output": out_safety["log"]}
    )
