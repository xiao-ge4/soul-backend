from __future__ import annotations
from typing import Optional, Dict, Any
from backend.models.types import PersonaState

# 简单进程内存（演示用）
_PERSONA_STATE = PersonaState(mbti=None, functions=None, enabled=False)


def get_persona_state() -> PersonaState:
	return _PERSONA_STATE


def apply_persona_state(mbti: Optional[str], functions: Optional[Dict[str, int]], enabled: bool) -> PersonaState:
	if mbti is not None:
		_PERSONA_STATE.mbti = mbti
	if functions is not None:
		# 归一范围
		_PERSONA_STATE.functions = {k: max(0, min(100, int(v))) for k, v in functions.items()}
	_PERSONA_STATE.enabled = bool(enabled)
	return _PERSONA_STATE


