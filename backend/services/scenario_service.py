from __future__ import annotations
from typing import Any, Dict, Optional, List

from backend.models.types import ScenarioInput, ScenarioContext, OpponentProfile, UserGoal, ScenarioFlow
from backend.clients.llm_client import analyze_scenario_llm


def _to_opponent(data: Dict[str, Any]) -> OpponentProfile:
	return OpponentProfile(
		roleTitle=(data.get("roleTitle") or None),
		style=(data.get("style") or None),
		tone=(data.get("tone") or None),
		traits=(data.get("traits") or None),
		domain=(data.get("domain") or None),
	)


def _to_user_goal(data: Dict[str, Any]) -> UserGoal:
	return UserGoal(
		goal=(data.get("goal") or None),
		subgoals=(data.get("subgoals") or None),
		successCriteria=(data.get("successCriteria") or None),
		priority=(data.get("priority") or None),
		reason=(data.get("reason") or None),
	)


def analyze_scenario(req: ScenarioInput) -> ScenarioContext:
	payload: Dict[str, Any] = {
		"templateId": req.templateId,
		"scenarioText": req.scenarioText,
		"opponentHint": req.opponentHint,
		"userGoalHint": req.userGoalHint,
		"mode": req.mode or "full",
		"opponentTraits": req.opponentTraits or None,
	}
	data = analyze_scenario_llm(payload) or {}
	if not isinstance(data, dict):
		data = {}

	scn_text = (data.get("scenario") or req.scenarioText or "")
	oppo = data.get("opponent") or {}
	ug = data.get("userGoal") or {}
	anchors = data.get("anchors") or None
	constraints = data.get("constraints") or None
	flow_data = data.get("flow") or {}

	# Fallbacks
	if not oppo:
		oppo = {}
	if not ug:
		ug = {"goal": req.userGoalHint or ""}

    # goal_only 模式下可能仅返回 userGoal
	flow_obj = None
	if isinstance(flow_data, dict):
		flow_obj = ScenarioFlow(
			startingParty=flow_data.get("startingParty") or "either",
			openingHints=flow_data.get("openingHints") if isinstance(flow_data.get("openingHints"), list) else None
		)
	return ScenarioContext(
		scenario=(scn_text or req.scenarioText or None),
		opponent=_to_opponent(oppo) if oppo else None,
		userGoal=_to_user_goal(ug),
		constraints=constraints if isinstance(constraints, dict) else None,
		anchors=anchors if isinstance(anchors, list) else None,
		flow=flow_obj
	)
