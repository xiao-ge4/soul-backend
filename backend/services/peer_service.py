from __future__ import annotations
from typing import Any, Dict, List

from backend.clients.llm_client import chat_completion
from backend.models.types import PeerReplyRequest, PeerReplyResponse


def generate_peer_reply(req: PeerReplyRequest) -> PeerReplyResponse:
	conv_list = [t.model_dump() for t in req.conversation][-12:]
	
	# 格式化对话历史
	conv_formatted = []
	for turn in conv_list:
		role = turn.get('role', 'unknown')
		text = turn.get('text', '')
		if role == 'user':
			conv_formatted.append(f"我（用户）：{text}")
		elif role == 'peer':
			conv_formatted.append(f"你（{turn.get('roleTitle', '对方')}）：{text}")
	conv_str = "\n".join(conv_formatted) if conv_formatted else "（无对话历史）"
	style = (req.opponent.style if req.opponent and req.opponent.style else "自然").strip()
	hint = (req.opponent.persona_hint if req.opponent and req.opponent.persona_hint else "").strip()
	role_title = (req.opponent.roleTitle if req.opponent and req.opponent.roleTitle else "").strip()
	tone = (req.opponent.tone if req.opponent and req.opponent.tone else "").strip()
	traits = (req.opponent.traits if req.opponent and req.opponent.traits else []) or []
	domain = (req.opponent.domain if req.opponent and req.opponent.domain else "").strip()

	scn_desc = ""
	if req.scenario:
		try:
			scn = req.scenario.model_dump()
			oppo = scn.get("opponent") or {}
			ug = scn.get("userGoal") or {}
			if oppo.get("style"):
				style = str(oppo.get("style")).strip() or style
			scn_desc = (
				f"场景：{scn.get('scenario') or ''}；领域：{oppo.get('domain') or domain}；"
				f"对方：{oppo.get('roleTitle') or role_title}，风格：{oppo.get('style') or style}，语气：{oppo.get('tone') or tone}，特征：{','.join(oppo.get('traits') or traits)}；"
				f"我的目标：{ug.get('goal') or ''}"
			)
		except Exception:
			scn_desc = ""

	sys = (
		"你是一位中文虚拟聊天对象，目标是自然地与对方交流。"
		"请根据你在场景中的身份和立场，使用符合该角色的语气、称谓和行为方式。"
	)
	style_map = {
		"自然": "语气自然、不做作，表达清楚即可。",
		"活泼": "语气轻快，偶尔用表情或拟声，加强互动感，但不过度。",
		"理性": "语气沉稳偏理性，简洁、有逻辑，适度反问推进话题。",
		"温和": "语气温柔与支持，给对方积极反馈与简短共情。",
		"专业": "语气专业、信息密度较高，但不说教，注意浅显表达。",
		"俏皮": "语气俏皮幽默，避免讽刺与刻板印象，轻松而不失礼貌。",
		"克制": "语气简洁克制，不热情但不冷漠，回应在点上。",
	}
	# 优先使用 traits 作为对方形象关键词；若存在，则以其为准
	if traits:
		style_desc = (
			f"请参考对方形象关键词：{'、'.join([str(t) for t in traits if t])}。"
			"优先依据这些关键词调整语气、关注点与说话方式；若与固定风格冲突，以关键词为准。"
		)
	else:
		style_desc = style_map.get(style, style_map["自然"])
	persona_hint = f"对手设定：{hint}。" if hint else ""
	role_hint = f"你的角色：{role_title}" if role_title else "你的角色：对话对象"
	# 获取最后一句用户说的话（如果存在）
	last_user_msg = ""
	if conv_list:
		for turn in reversed(conv_list):
			if turn.get('role') == 'user':
				last_user_msg = turn.get('text', '')
				break
	
	usr = (
		f"请扮演与我聊天的对象，风格：{style}（{style_desc}）。{persona_hint}\n"
		f"{role_hint}\n\n"
		f"{('场景设定：' + scn_desc + '\n') if scn_desc else ''}"
		"\n【对话历史】\n"
		f"{conv_str}\n\n"
		"【回复要求】\n"
		f"对方（用户）最后一句话是：{last_user_msg if last_user_msg else '（无）'}\n"
		"你必须针对这句话给出直接、相关的回复。\n\n"
		"重要规则：\n"
		"1. 中文输出，每条不超过2句\n"
		"2. 不要重复问已经回答过的问题（如果对方已经解释了某事，不要再问）\n"
		"3. 如果对方提出邀请或问你是否有兴趣，应该回应是/否，而不是反问\n"
		f"4. 理解你的身份定位：作为{role_title}，应结合场景和对话历史决定合适的主动或被动程度\n"
		f"5. 场景逻辑：作为{role_title}，不要说不符合身份的话（如学弟不会说'我们社团'，应该说'你们社团'）\n\n"
		"请以 JSON 数组返回 3 条不同态度的回复（积极/中立/委婉拒绝）。\n"
		"格式：[{\"id\":\"pos\",\"text\":\"...\",\"tone\":\"positive\"},{\"id\":\"neut\",\"text\":\"...\",\"tone\":\"neutral\"},{\"id\":\"neg\",\"text\":\"...\",\"tone\":\"negative\"}]\n"
		"只输出 JSON 数组，不要任何解释文字。"
	)

	try:
		raw = chat_completion(
			[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
			max_tokens=300,
			temperature=0.8,
		).strip()
	except Exception:
		raw = ""

	from backend.clients.llm_client import _safe_json_parse
	from backend.models.types import PeerReplyItem
	
	data = _safe_json_parse(raw) or []
	replies = []
	if isinstance(data, list):
		for item in data[:3]:
			if isinstance(item, dict) and item.get("text"):
				replies.append({
					"id": item.get("id", "alt"),
					"text": str(item.get("text")),
					"tone": item.get("tone"),
					"why": item.get("why")
				})
	
	if not replies:
		# fallback
		if raw and isinstance(raw, str):
			replies = [{"id": "default", "text": raw}]
		else:
			replies = [{"id": "default", "text": "我们可以继续聊聊刚才的话题～你怎么看？"}]
	
	return PeerReplyResponse(
		text=replies[0]["text"],
		replies=[PeerReplyItem(**r) for r in replies]
	)


