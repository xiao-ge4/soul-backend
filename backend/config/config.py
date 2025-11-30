from pathlib import Path
from typing import Optional
import os
from openai import OpenAI

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN_FILE = BASE_DIR / "config" / "modelscope_token.txt"

# Model & endpoint configuration
MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-8B")
BASE_URL = os.getenv("MODEL_BASE_URL", "https://api-inference.modelscope.cn/v1")

# Common env var names used by不同平台（取其一即可）
_TOKEN_ENV_CANDIDATES = [
	"MODELSCOPE_TOKEN",
	"MODELSCOPE_API_TOKEN",
	"MODELSCOPE_API_KEY",
	"MSPACE_TOKEN",
	"MSPACE_API_TOKEN",
	"MSPACE_API_KEY",
	"OPENAI_API_KEY",  # 兼容部分统一Key名
]


def read_modelscope_token() -> str:
	"""
	Read ModelScope token from env or token file.
	优先读取环境变量，其次读取 backend/config/modelscope_token.txt。
	"""
	for name in _TOKEN_ENV_CANDIDATES:
		val = os.getenv(name)
		if val:
			return val.strip()
	if TOKEN_FILE.exists():
		return TOKEN_FILE.read_text(encoding="utf-8").strip()
	candidates = ", ".join(_TOKEN_ENV_CANDIDATES)
	raise RuntimeError(
		f"未检测到模型 Token。请在环境变量中设置其一：[{candidates}]，"
		f"或在 {TOKEN_FILE} 写入 Token。"
	)


def create_openai_client() -> OpenAI:
	"""
	Create OpenAI-compatible client for ModelScope/Qwen.
	"""
	return OpenAI(
		base_url=BASE_URL,
		api_key=read_modelscope_token(),
	)


