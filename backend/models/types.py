from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
	role: Literal["user", "peer"]
	text: str
	ts: Optional[float] = None


class Profile(BaseModel):
	interests: Optional[List[str]] = None
	bio: Optional[str] = None
	stylePref: Optional[str] = None


class MemoryItem(BaseModel):
	type: Literal["wish", "preference", "note"] = "note"
	content: str


class PersonaWeights(BaseModel):
	Ni: int = 0
	Ne: int = 0
	Si: int = 0
	Se: int = 0
	Ti: int = 0
	Te: int = 0
	Fi: int = 0
	Fe: int = 0
	enabled: bool = False


class SuggestRequest(BaseModel):
	conversation: List[ConversationTurn]
	draft: Optional[str] = ""
	entryType: Literal["typing", "preSend", "postSend", "peerMsg", "idle", "firstEnter"] = "typing"
	userProfile: Optional[Profile] = None
	peerProfile: Optional[Profile] = None
	memory: Optional[List[MemoryItem]] = None
	personaWeights: Optional[PersonaWeights] = None
	scenario: Optional["ScenarioContext"] = None


class Tip(BaseModel):
	text: str
	tone: Literal["gentle", "neutral", "alert"] = "gentle"
	risk: Literal["very_low", "low", "mid", "high"] = "low"


class Candidate(BaseModel):
	id: str = Field(default="cand")
	text: str
	why: str = ""
	risk: Literal["low", "mid", "high"] = "low"
	score: float = 0.0


class Relationship(BaseModel):
	index: int = 50
	trend: Literal["up", "flat", "down"] = "flat"


class Safety(BaseModel):
	blocked: bool = False
	notes: List[str] = []


class SuggestResponse(BaseModel):
	tip: Tip
	candidates: List[Candidate]
	relationship: Relationship
	safety: Safety


class MBTIAnswer(BaseModel):
	dim: Literal["EI", "SN", "TF", "JP"]
	value: int = Field(ge=1, le=5)
	reverse: bool = False


class MBTISubmitRequest(BaseModel):
	answers: List[MBTIAnswer]
	mode: Literal["quick", "deep"] = "quick"


class MBTISubmitResponse(BaseModel):
	mbti: str
	confidence: float
	functions: Dict[str, int]
	advice: List[str]


class MBTIInferRequest(BaseModel):
	conversation: List[ConversationTurn]


class MBTIInferResponse(BaseModel):
	mbtiGuess: str
	confidence: float
	functionsGuess: Dict[str, int]
	notes: str = ""


class PersonaState(BaseModel):
	mbti: Optional[str] = None
	functions: Optional[Dict[str, int]] = None
	enabled: bool = False


class OpponentProfile(BaseModel):
	style: Optional[str] = "自然"  # 自然/活泼/理性/温和/专业/俏皮/克制
	persona_hint: Optional[str] = None  # 练习目标或设定（如“新认识、爱徒步”）
	roleTitle: Optional[str] = None
	traits: Optional[List[str]] = None
	domain: Optional[str] = None
	tone: Optional[str] = None


class PeerReplyRequest(BaseModel):
	conversation: List[ConversationTurn]
	opponent: Optional[OpponentProfile] = None
	personaWeights: Optional[PersonaWeights] = None  # 可选，用于影响对手理解用户偏好
	scenario: Optional["ScenarioContext"] = None


class PeerReplyItem(BaseModel):
	id: str
	text: str
	tone: Optional[str] = None
	why: Optional[str] = None


class PeerReplyResponse(BaseModel):
	text: str  # 兼容旧字段，取第一条回复
	replies: Optional[List[PeerReplyItem]] = None


class UserGoal(BaseModel):
	goal: Optional[str] = None
	subgoals: Optional[List[str]] = None
	successCriteria: Optional[List[str]] = None
	priority: Optional[str] = None
	reason: Optional[str] = None


class ScenarioFlow(BaseModel):
	startingParty: Optional[Literal["user", "opponent", "either"]] = "either"
	openingHints: Optional[List[str]] = None


class ScenarioContext(BaseModel):
	scenario: Optional[str] = None
	opponent: Optional[OpponentProfile] = None
	userGoal: Optional[UserGoal] = None
	constraints: Optional[Dict[str, Any]] = None
	anchors: Optional[List[str]] = None
	flow: Optional[ScenarioFlow] = None


class ScenarioInput(BaseModel):
	templateId: Optional[str] = None
	scenarioText: Optional[str] = None
	opponentHint: Optional[str] = None
	userGoalHint: Optional[str] = None
	mode: Optional[Literal["full", "goal_only"]] = "full"
	opponentTraits: Optional[List[str]] = None
