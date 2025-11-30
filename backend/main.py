from __future__ import annotations
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.models.types import (
	SuggestRequest, SuggestResponse,
	MBTISubmitRequest, MBTISubmitResponse,
	MBTIInferRequest, MBTIInferResponse,
	PersonaState,
	PeerReplyRequest, PeerReplyResponse,
	ScenarioInput, ScenarioContext
)
from backend.services.suggest_service import handle_suggest
from backend.services.persona_service import compute_mbti_submit
from backend.clients.llm_client import infer_mbti_from_chat
from backend.services.memory_service import get_persona_state, apply_persona_state
from backend.services.peer_service import generate_peer_reply
from backend.services.scenario_service import analyze_scenario

app = FastAPI(title="Soul-Agent Demo", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# API
@app.post("/api/suggest", response_model=SuggestResponse)
def api_suggest(req: SuggestRequest):
	return handle_suggest(req)


@app.post("/api/mbti/submit", response_model=MBTISubmitResponse)
def api_mbti_submit(req: MBTISubmitRequest):
	return compute_mbti_submit(req)


@app.post("/api/mbti/infer-from-chat", response_model=MBTIInferResponse)
def api_mbti_infer_from_chat(req: MBTIInferRequest):
	data = infer_mbti_from_chat([t.model_dump() for t in req.conversation])
	return MBTIInferResponse(
		mbtiGuess=data.get("mbti") or "",
		confidence=float(data.get("confidence", 0.0)),
		functionsGuess=data.get("functions") or {},
		notes=data.get("notes") or "",
	)


@app.get("/api/persona", response_model=PersonaState)
def api_get_persona():
	return get_persona_state()


@app.post("/api/persona/apply", response_model=PersonaState)
def api_apply_persona(state: PersonaState):
	return apply_persona_state(state.mbti, state.functions, state.enabled)

@app.post("/api/peer/reply", response_model=PeerReplyResponse)
def api_peer_reply(req: PeerReplyRequest):
	return generate_peer_reply(req)


# 场景分析
@app.post("/api/scenario/analyze", response_model=ScenarioContext)
def api_scenario_analyze(req: ScenarioInput):
	return analyze_scenario(req)


# 静态资源（前端）
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


