from __future__ import annotations
from typing import Dict, List, Tuple
from statistics import mean

from backend.models.types import MBTISubmitRequest, MBTISubmitResponse


def _score_dim(answers: List[Tuple[int, bool]]) -> float:
	"""
	answers: list of (value 1..5, reverse)
	return normalized score 0..1 (towards first letter)
	"""
	if not answers:
		return 0.5
	values = []
	for v, rev in answers:
		val = int(v)
		val = max(1, min(5, val))
		if rev:
			val = 6 - val
		values.append(val)
	# map 1..5 -> 0..1
	return (mean(values) - 1.0) / 4.0


def _pick_letter(score: float, first: str, second: str) -> Tuple[str, float]:
	conf = abs(score - 0.5) * 2  # 0..1
	return (first if score >= 0.5 else second, conf)


def _functions_from_mbti(mbti: str) -> Dict[str, int]:
	"""
	Simple heuristic mapping to Jung functions default weights (0..100).
	"""
	mbti = (mbti or "").upper()
	default = {"Ni":10,"Ne":10,"Si":10,"Se":10,"Ti":10,"Te":10,"Fi":10,"Fe":10}
	stack_map = {
		"INTJ": ["Ni","Te","Fi","Se"],
		"ENTJ": ["Te","Ni","Se","Fi"],
		"INFJ": ["Ni","Fe","Ti","Se"],
		"ENFJ": ["Fe","Ni","Se","Ti"],
		"INTP": ["Ti","Ne","Si","Fe"],
		"ENTP": ["Ne","Ti","Fe","Si"],
		"INFP": ["Fi","Ne","Si","Te"],
		"ENFP": ["Ne","Fi","Te","Si"],
		"ISTJ": ["Si","Te","Fi","Ne"],
		"ESTJ": ["Te","Si","Ne","Fi"],
		"ISFJ": ["Si","Fe","Ti","Ne"],
		"ESFJ": ["Fe","Si","Ne","Ti"],
		"ISTP": ["Ti","Se","Ni","Fe"],
		"ESTP": ["Se","Ti","Fe","Ni"],
		"ISFP": ["Fi","Se","Ni","Te"],
		"ESFP": ["Se","Fi","Te","Ni"],
	}
	stack = stack_map.get(mbti, [])
	weights = [35, 25, 15, 10]
	funcs = dict(default)
	for i, f in enumerate(stack):
		funcs[f] = weights[i]
	# shadow (very rough)
	for f in funcs:
		if f not in stack:
			funcs[f] = min(funcs[f], 15)
	total = sum(funcs.values())
	# normalize to max 100 (already near)
	return {k: int(v) for k, v in funcs.items()}


def compute_mbti_submit(req: MBTISubmitRequest) -> MBTISubmitResponse:
	by_dim = {"EI": [], "SN": [], "TF": [], "JP": []}
	for a in req.answers:
		by_dim[a.dim].append((a.value, a.reverse))
	scores = {k: _score_dim(v) for k, v in by_dim.items()}
	ei, ei_c = _pick_letter(scores["EI"], "E", "I")
	sn, sn_c = _pick_letter(scores["SN"], "S", "N")
	tf, tf_c = _pick_letter(scores["TF"], "T", "F")
	jp, jp_c = _pick_letter(scores["JP"], "J", "P")
	mbti = f"{ei}{sn}{tf}{jp}"
	conf = float(round((ei_c + sn_c + tf_c + jp_c) / 4.0, 2))
	funcs = _functions_from_mbti(mbti)
	advice = []
	# brief advice based on S/N & T/F
	if sn == "S":
		advice.append("偏好具体与实例，沟通时给出可执行的小步骤")
	else:
		advice.append("偏好愿景与类比，沟通时给出整体框架")
	if tf == "T":
		advice.append("偏好逻辑与事实，避免情绪化措辞")
	else:
		advice.append("偏好感受与价值，表达共情更易被接受")
	return MBTISubmitResponse(mbti=mbti, confidence=conf, functions=funcs, advice=advice)


