from __future__ import annotations
import re
from typing import Dict, List

_BANNED = {
	"仇恨", "歧视", "辱骂", "约炮", "涉黄", "黄赌毒", "极端", "恐怖",
}

_PII_PATTERNS = [
	re.compile(r"\b1[3-9]\d{9}\b"),  # 简单手机
	re.compile(r"\b\d{17}[\dxX]\b"),  # 简单身份证
]


def safety_check_text(text: str) -> Dict:
	notes: List[str] = []
	lower = text.strip()
	blocked = False
	for w in _BANNED:
		if w in lower:
			notes.append(f"包含敏感词: {w}")
			blocked = True
	for p in _PII_PATTERNS:
		if p.search(lower):
			notes.append("疑似包含个人敏感信息")
	return {"blocked": blocked, "notes": notes}


def redact_if_needed(text: str) -> str:
	t = text
	for p in _PII_PATTERNS:
		t = p.sub("[已脱敏]", t)
	return t


