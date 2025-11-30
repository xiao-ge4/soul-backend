from __future__ import annotations
from typing import Any, Dict, List, Optional
from statistics import mean

from backend.clients.llm_client import generate_candidates
from backend.models.types import (
	SuggestRequest, SuggestResponse, Tip, Candidate, Relationship, Safety
)
from backend.services.safety_service import safety_check_text, redact_if_needed

_POS_WORDS = {"喜欢", "开心", "有趣", "好玩", "期待", "不错", "赞", "哈哈", "开心"}
_NEG_WORDS = {"无聊", "烦", "不想", "不愿", "生气", "晚回", "算了", "唉"}


def _extract_keywords(text: str) -> list[str]:
	"""
	极简关键词抽取：按常见分隔符切分，保留长度>=2的片段，去重后取前5个。
	避免引入第三方分词库，足以用于“上下文锚点”。
	"""
	if not text:
		return []
	seps = "，。！？!?,.;:：、()（）[]【】<>《》\"' \n\t"
	tmp = text
	for ch in seps:
		tmp = tmp.replace(ch, " ")
	parts = [p.strip() for p in tmp.split(" ") if p.strip()]
	parts = [p for p in parts if len(p) >= 2]
	# 去重保持顺序
	seen = set()
	result = []
	for p in parts:
		if p not in seen:
			result.append(p)
			seen.add(p)
	return result[:5]


def _affect_score(text: str) -> float:
	score = 0
	for w in _POS_WORDS:
		if w in text:
			score += 1
	for w in _NEG_WORDS:
		if w in text:
			score -= 1
	return max(-3, min(3, score)) / 3.0


def _analyze_conversation(conv: List[Dict[str, Any]]) -> Dict[str, Any]:
	peer_texts = [t["text"] for t in conv if t.get("role") == "peer" and t.get("text")]
	aff = mean([_affect_score(t) for t in peer_texts[-5:]]) if peer_texts else 0.0
	relationship_index = int(round(50 + aff * 30))
	relationship_index = max(0, min(100, relationship_index))
	# 判断最后一条是否为对方问句
	last_peer_is_question = False
	last_role = conv[-1]["role"] if conv else None
	last_text = conv[-1]["text"] if conv else ""
	if last_role == "peer" and last_text:
		if last_text.strip().endswith(("?", "？")):
			last_peer_is_question = True
	anchor_keywords = _extract_keywords(last_text if last_role == "peer" else "")
	return {
		"affect": aff,
		"relationship_index": relationship_index,
		"trend": "up" if aff > 0.15 else ("down" if aff < -0.15 else "flat"),
		"last_peer_is_question": last_peer_is_question,
		"last_role": last_role,
		"last_text": last_text,
		"anchor_keywords": anchor_keywords,
	}


def _build_tip(analysis: Dict[str, Any], entry_type: str, draft: str) -> Tip:
	aff = analysis["affect"]
	if analysis.get("last_peer_is_question"):
		# 若对方刚提问，优先提醒“先回答再补一句”
		if draft:
			return Tip(text="先回答TA的问题，再补一个小细节", tone="gentle", risk="low")
		return Tip(text="建议先答再问：给出你的看法或经历", tone="gentle", risk="low")
	if entry_type in ("preSend", "typing") and draft:
		if aff < -0.2:
			return Tip(text="建议降低强度，先共情再提问", tone="alert", risk="mid")
		if len(draft) < 8:
			return Tip(text="建议更具体些，给出一个小细节", tone="gentle", risk="low")
		return Tip(text="保持自然语气，附带一个轻问题", tone="gentle", risk="very_low")
	if entry_type in ("idle",):
		return Tip(text="尝试承接TA的兴趣点，给一个续聊锚点", tone="neutral", risk="low")
	return Tip(text="继续保持节奏～", tone="gentle", risk="very_low")


def _score_candidate(text: str, why: str, risk: str, analysis: Dict[str, Any]) -> float:
	base = 0.5
	q_count = text.count("？") + text.count("?")
	if analysis.get("last_peer_is_question"):
		# 对方刚提问：回答优先，减少继续发问
		if q_count == 0:
			base += 0.12
		elif q_count == 1:
			base += 0.02
		else:
			base -= 0.08
	else:
		if q_count >= 1:
			base += 0.1  # 促进互动
	# 主题锚点重合度
	anchors = analysis.get("anchor_keywords") or []
	if anchors:
		match = sum(1 for k in anchors if k and k in text)
		if analysis.get("last_peer_is_question"):
			if match >= 1:
				base += 0.15
			else:
				base -= 0.12
		else:
			if match >= 1:
				base += 0.06
	sc_keys = analysis.get("scenario_keywords") or []
	if sc_keys:
		m2 = sum(1 for k in sc_keys if k and k in text)
		if m2 >= 1:
			base += 0.06
	if len(text) <= 40:
		base += 0.05  # 简洁
	if risk == "low":
		base += 0.08
	if analysis["affect"] < -0.2 and "幽默" in why:
		base -= 0.05  # 负面时降低幽默权重
	return max(0.0, min(1.0, base))


def _fallback_from_context(conv: List[Dict[str, Any]], draft: str, reply_mode: str) -> List[Dict[str, str]]:
	"""当模型超时/限流时的本地候选兜底（面向“你将要发送”的下一条）。"""
	last_peer = ""
	last_user = ""
	last_role = None
	for t in reversed(conv):
		if t.get("role") in ("user", "peer"):
			last_role = t["role"]
			break
	if last_role == "peer":
		for t in reversed(conv):
			if t.get("role") == "peer" and t.get("text"):
				last_peer = t["text"]
				break
	else:
		for t in reversed(conv):
			if t.get("role") == "user" and t.get("text"):
				last_user = t["text"]
				break

	# 若草稿已存在：做“增强与收束”
	if draft:
		return [
			{"id":"mirror","text":f"{draft} 想听听你的看法～","why":"承接草稿并抛球","risk":"low"},
			{"id":"safe","text":"我先说到这里，你这边怎么看？","why":"稳妥推进","risk":"low"},
			{"id":"humor","text":"这段我就不剧透啦，交给你来补完？😄","why":"轻松化","risk":"mid"},
		]

	# 最近一条为对方消息：承接对方
	if last_role == "peer":
		if reply_mode == "answer":
			return [
				{"id":"mirror","text":f"我这边主要是{last_peer[:10]}这部分的体验比较深～如果你想我可以具体说说。","why":"先回答再补充","risk":"low"},
				{"id":"safe","text":"我的看法是这样……（简单两点）如果你也方便，想听听你的想法。","why":"给出答案+轻抛球","risk":"low"},
				{"id":"humor","text":"先交一份简短答卷，再抛个小问题：你会怎么选？","why":"回答后轻松推进","risk":"mid"},
			]
		return [
			{"id":"mirror","text":f"关于“{last_peer[:18]}”，你更在意哪一部分？","why":"承接其话题","risk":"low"},
			{"id":"safe","text":"如果方便的话，能说说具体是怎么想的吗？","why":"稳妥追问","risk":"low"},
			{"id":"humor","text":"不如来个快问快答，我先抛一个：你会选A还是B？","why":"轻松推进","risk":"mid"},
		]

	# 最近一条为我方消息：做“自我补充 + 抛回对方”
	if last_role == "user":
		return [
			{"id":"mirror","text":"主要是我这次在某一科状态更好～你最近有什么小高光？","why":"自述+抛回","risk":"low"},
			{"id":"safe","text":"我的部分先到这儿，你这边最近有什么想分享的吗？","why":"稳妥转问","risk":"low"},
			{"id":"humor","text":"给自己发一张小小“表扬券”，也想听听你的故事～","why":"轻松转场","risk":"mid"},
		]

	# 默认开场
	return [
		{"id":"mirror","text":"周末一般怎么放松？我最近迷上了散步。","why":"开启轻话题","risk":"low"},
		{"id":"safe","text":"不急，我们可以从兴趣开始聊起～","why":"稳妥开场","risk":"low"},
		{"id":"humor","text":"发你一张“聊天启动券”，换你一个小分享？","why":"幽默破冰","risk":"mid"},
	]


def handle_suggest(req: SuggestRequest) -> SuggestResponse:
	conv = [t.model_dump() for t in req.conversation]
	analysis = _analyze_conversation(conv)
	scenario_keywords: List[str] = []
	if req.scenario and req.scenario.anchors:
		scenario_keywords.extend([s for s in req.scenario.anchors if s])
	if req.scenario and req.scenario.userGoal and req.scenario.userGoal.goal:
		scenario_keywords.extend(_extract_keywords(req.scenario.userGoal.goal))
	analysis["scenario_keywords"] = scenario_keywords[:6]

	# 判断是否应由对方先开场
	starting_party = "either"
	if req.scenario and req.scenario.flow and req.scenario.flow.startingParty:
		starting_party = req.scenario.flow.startingParty
	if not conv and starting_party == "opponent":
		# 会话为空且应由对方先开场，不返回可发送候选
		tip = Tip(text="当前场景通常由对方先开场，请等待对方发起对话或点击“对方回复”。", tone="neutral", risk="low")
		rel = Relationship(index=analysis["relationship_index"], trend=analysis["trend"])
		safety = Safety(blocked=False, notes=[])
		return SuggestResponse(tip=tip, candidates=[], relationship=rel, safety=safety)

	tip = _build_tip(analysis, req.entryType, req.draft or "")

	context = {
		"conversation": conv[-12:],
		"draft": req.draft or "",
		"userProfile": (req.userProfile or {}).model_dump() if req.userProfile else {},
		"peerProfile": (req.peerProfile or {}).model_dump() if req.peerProfile else {},
		"anchor": {
			"last_role": analysis.get("last_role"),
			"last_text": analysis.get("last_text"),
			"keywords": analysis.get("anchor_keywords"),
		},
		"scenario": req.scenario.model_dump() if req.scenario else None,
	}
	persona = None
	if req.personaWeights:
		persona = {"enabled": req.personaWeights.enabled, "functions": req.personaWeights.model_dump()}
	if persona and "enabled" in persona["functions"]:
		persona["functions"].pop("enabled", None)

	try:
		reply_mode = "answer" if analysis.get("last_peer_is_question") else "probe"
		raw_cands = generate_candidates(context, persona=persona, reply_mode=reply_mode)
	except Exception:
		reply_mode = "answer" if analysis.get("last_peer_is_question") else "probe"
		raw_cands = _fallback_from_context(conv[-12:], req.draft or "", reply_mode)

	# 4) 安全审校、打分
	final_cands: List[Candidate] = []
	for it in raw_cands:
		safe = safety_check_text(it["text"])
		if safe["blocked"]:
			continue
		risk_val = str(it.get("risk", "low"))
		if risk_val not in ("low","mid","high"):
			risk_val = "low"
		score = _score_candidate(it["text"], it.get("why", ""), risk_val, analysis)
		final_cands.append(Candidate(
			id=it.get("id", "cand"),
			text=redact_if_needed(it["text"]),
			why=it.get("why", ""),
			risk=risk_val,
			score=score
		))

	# 最多取3条
	final_cands = sorted(final_cands, key=lambda x: x.score, reverse=True)[:3] or [
		Candidate(id="safe", text="不急～可以聊聊你最近在忙什么？", why="稳妥推进", risk="very_low", score=0.7)
	]

	rel = Relationship(index=analysis["relationship_index"], trend=analysis["trend"])
	safety = Safety(blocked=False, notes=[])
	return SuggestResponse(tip=tip, candidates=final_cands, relationship=rel, safety=safety)


