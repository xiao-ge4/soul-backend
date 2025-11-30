from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json

from backend.config.config import create_openai_client, MODEL_NAME

# Create a single client instance
_client = create_openai_client()


def chat_completion(
	messages: List[Dict[str, str]],
	max_tokens: int = 512,
	temperature: float = 0.6,
	extra_body: Optional[Dict[str, Any]] = None,
	use_stream: bool = False,
) -> str:
	"""
	Call ModelScope OpenAI-compatible chat completion and return content text.
	- 注意：ModelScope 的 enable_thinking 仅支持 stream 模式。
	"""
	kwargs: Dict[str, Any] = dict(
		model=MODEL_NAME,
		messages=messages,
		max_tokens=max_tokens,
		temperature=temperature,
	)
	if use_stream:
		kwargs["stream"] = True
		kwargs["extra_body"] = extra_body or {"enable_thinking": True}
	else:
		kwargs["stream"] = False
		# 关键：非流式需明确关闭 thinking
		kwargs["extra_body"] = {"enable_thinking": False}
	resp = _client.chat.completions.create(**kwargs)
	content = resp.choices[0].message.content or ""
	return content


def _safe_json_parse(text: str) -> Any:
	try:
		return json.loads(text)
	except Exception:
		# Try to extract the first JSON block
		start = text.find("{")
		brack = text.find("[")
		if brack != -1 and (start == -1 or brack < start):
			start = brack
		if start != -1:
			frag = text[start:]
			# Try trimming trailing non-json
			for end in range(len(frag), max(len(frag) - 4000, 0), -1):
				try:
					return json.loads(frag[:end])
				except Exception:
					continue
		return None


def safe_json_parse(text: str) -> Any:
	return _safe_json_parse(text)


def generate_candidates(
	context: Dict[str, Any],
	persona: Optional[Dict[str, Any]] = None,
	reply_mode: str = "probe",  # "answer" | "probe"
) -> List[Dict[str, Any]]:
	"""
	Use LLM to generate 3+ candidate replies (mirror/safe/humor),
	then caller can score and pick top-3.
	"""
	persona_hint = ""
	if persona and persona.get("enabled"):
		funcs = persona.get("functions") or {}
		persona_hint = f"\n已知用户八维偏好：{json.dumps(funcs, ensure_ascii=False)}。请尽量匹配沟通风格。"

	sys = (
		"你是一位中文沟通教练助手，专门帮助用户提升社交对话技巧。"
		"你的任务是为用户生成多条候选回复，帮助用户学习如何更好地与对方沟通。"
	)
	mode_hint = ""
	if reply_mode == "answer":
		mode_hint = (
			"\n当前应对模式：answer（对方刚提出问题）。"
			"\n请先直接给出回答/信息/观点，不要以提问开头；整条最多可包含0-1个轻问（可为0）。"
			"\n尽量具体，结合上下文中的事实或常识补充一个小细节，再视情况加一句轻提问。"
		)
	else:
		mode_hint = (
			"\n当前应对模式：probe（推进对话）。"
			"\n可以包含一个自然追问，用于推动互动。"
		)
	scenario = context.get("scenario") or {}
	scenario_hint = ""
	if scenario:
		oppo = scenario.get("opponent") or {}
		ug = scenario.get("userGoal") or {}
		traits = oppo.get("traits") or []
		scn_desc = {
			"scenario": scenario.get("scenario") or "",
			"opponent": {
				"roleTitle": oppo.get("roleTitle") or "",
				"style": oppo.get("style") or "",
				"tone": oppo.get("tone") or "",
				"traits": traits,
				"domain": oppo.get("domain") or "",
			},
			"userGoal": {
				"goal": (ug.get("goal") or ""),
				"subgoals": ug.get("subgoals") or [],
				"successCriteria": ug.get("successCriteria") or [],
			},
		}
		traits_hint = ("；对方形象关键词：" + "、".join([str(t) for t in traits if t])) if traits else ""
		style_rule = (
			"\n请优先依据对方形象关键词调整语气、关注点与说话方式；"
			"若关键词与固定风格冲突，以关键词为准；避免与其相悖的表达。"
			if traits else ""
		)
		role_title = scn_desc['opponent']['roleTitle'] or '对方'
		scenario_desc_text = scn_desc.get('scenario') or ''
		user_goal = scn_desc['userGoal']['goal'] or '自然交流'
		
		scenario_hint = (
			f"\n场景设定：{json.dumps(scn_desc, ensure_ascii=False)}{traits_hint}。\n\n"
			"【极其重要的身份逻辑】\n"
			"根据场景描述和对方角色，你需要推断出用户的身份。\n"
			f"场景描述：{scenario_desc_text}\n"
			f"对方角色：{role_title}\n"
			f"用户目标：{user_goal}\n\n"
			"基于以上信息，请明确：\n"
			"1. 用户的身份是什么？（例如：如果对方是学弟且场景是社团招新，那用户就是学长/学姐；如果对方是面试官，用户就是求职者）\n"
			"2. 用户和对方的关系是什么？（引导者vs被引导者？平等关系？）\n"
			"3. 用户在这个场景中的角色定位是什么？\n\n"
			"【候选生成要求】\n"
			"你是为“用户”（而不是对方）生成候选回复。\n"
			"候选回复必须：\n"
			"1. 以用户的真实身份口吻说话（根据你的推断）\n"
			f"2. 适合对{role_title}说的话\n"
			"3. 符合场景逻辑和社交常识（例如：社团成员介绍自己社团说'我们'，不说'你们'；求职者回答问题，不反问面试官的个人兴趣）\n"
			"4. 推进用户目标的实现\n\n"
			"【举例说明】\n"
			"错误示例：如果用户是学长招新，说'听说你们社团很有趣'←这是学弟的口吻\n"
			"正确示例：学长招新应说'我们社团最近有个活动很有趣'←这才是学长的口吻\n"
			+ style_rule
		)

	usr = (
		"请基于提供的对话上下文与画像，输出3-6条中文候选回复，槽位包含：镜像/稳妥/幽默。"
		"\n要求：每条≤2句；避免冒犯、隐私、刻板印象。"
		f"{mode_hint}"
		"\n如果上一条是对方消息，请优先引用上一条中的关键词或关键短语，保持紧密承接；若无法引用请说明原因再简洁回应。"
		"\n上下文锚点（可能为空）："
		f"{json.dumps(context.get('anchor', {}), ensure_ascii=False)}"
		"\n输出严格为JSON数组：[{\"id\":\"mirror|safe|humor|...\",\"text\":\"...\",\"why\":\"原因\",\"risk\":\"low|mid|high\"}]\n"
		f"上下文：{json.dumps(context, ensure_ascii=False)}"
		f"{scenario_hint}"
		f"{persona_hint}"
	)
	raw = chat_completion(
		[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
		max_tokens=512,
		temperature=0.7,
	)
	data = _safe_json_parse(raw)
	if not isinstance(data, list):
		return []
	cands = []
	for it in data:
		if not isinstance(it, dict):
			continue
		text = (it.get("text") or "").strip()
		if not text:
			continue
		cands.append({
			"id": it.get("id") or "cand",
			"text": text,
			"why": it.get("why") or "",
			"risk": it.get("risk") or "low"
		})
	return cands


def infer_mbti_from_chat(messages_for_infer: List[Dict[str, str]]) -> Dict[str, Any]:
	"""
	Use LLM to infer MBTI and Jung functions with confidence.
	"""
	sys = "你是性格与沟通风格分析助手。"
	usr = (
		"基于以下中文聊天记录，推断说话者（第一人称）的MBTI与荣格八维强度（0-100）。"
		"\n请给出证据点（抽象/具体、情感词密度、疑问/推理词、直接/委婉等），"
		"\n仅输出JSON对象：{"
		"\"mbti\":\"INTJ\","
		"\"confidence\":0.0,"
		"\"functions\":{\"Ni\":0,\"Ne\":0,\"Si\":0,\"Se\":0,\"Ti\":0,\"Te\":0,\"Fi\":0,\"Fe\":0},"
		"\"notes\":\"简要证据\"}"
		"\n聊天记录："
		f"{json.dumps(messages_for_infer, ensure_ascii=False)}"
	)
	raw = chat_completion(
		[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
		max_tokens=400,
		temperature=0.2,
	)
	data = _safe_json_parse(raw) or {}
	if not isinstance(data, dict):
		data = {}
	# normalize
	funcs = data.get("functions") or {}
	for k in ["Ni","Ne","Si","Se","Ti","Te","Fi","Fe"]:
		try:
			funcs[k] = max(0, min(100, int(funcs.get(k, 0))))
		except Exception:
			funcs[k] = 0
	data["functions"] = funcs
	data["mbti"] = (data.get("mbti") or "").upper()[:4]
	try:
		data["confidence"] = float(data.get("confidence", 0.0))
	except Exception:
		data["confidence"] = 0.0
	return data


def analyze_scenario_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
	mode = (payload.get("mode") or "full").lower()
	sys = "你是沟通教练助手，负责将自然语言的场景与意图结构化为可执行的沟通设定。"
	if mode == "goal_only":
		schema = "{\"userGoal\":{\"goal\":\"\",\"reason\":\"\"}}"
		guide = (
			"仅根据给定的‘场景描述’与‘对方形象关键词’推断并精炼一个适合当前轮次的沟通目标，"
			"用简洁中文表达；必要时给出形成该目标的‘reason’（一句话）。"
		)
	else:
		schema = (
			"{\"scenario\":\"...\","
			"\"opponent\":{\"roleTitle\":\"\",\"tone\":\"\",\"traits\":[],\"domain\":\"\"},"
			"\"userGoal\":{\"goal\":\"\",\"reason\":\"\",\"subgoals\":[],\"successCriteria\":[]},"
			"\"flow\":{\"startingParty\":\"user|opponent|either\",\"openingHints\":[]},"
			"\"anchors\":[],\"constraints\":{\"taboo\":[],\"lengthHint\":\"\",\"askRatio\":\"\"}}"
		)
		guide = (
			"请从‘场景描述/模板’中抽象出简洁的对方形象关键词（3-6条短语，避免单字或空泛词），"
			"补全对方称谓/语气与可选领域；根据‘对方形象+场景’产出‘我的目标’，并给出简短reason。"
			"同时判断该场景通常由谁先开场：user(我方主动)/opponent(对方先说，如面试官、客服)/either(均可)，"
			"并在flow.openingHints中给出1-2条开场建议（若startingParty=opponent则给对方开场示例，否则给我方）。"
		)
	usr = (
		"严格按以下JSON Schema输出，不要添加解释：" + schema + "\n"
		+ guide + "\n输入：" + json.dumps(payload, ensure_ascii=False)
	)
	raw = chat_completion([
		{"role": "system", "content": sys},
		{"role": "user", "content": usr},
	], max_tokens=600, temperature=0.3)
	data = _safe_json_parse(raw) or {}
	if not isinstance(data, dict):
		data = {}
	return data
