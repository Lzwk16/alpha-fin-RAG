"""Settings: config.yaml (committed, non-secret) + environment variables / .env (secrets, overrides).

Three LLM endpoints, deliberately from different model families (same-model bias: a model grading questions it
wrote, or answers it would write, scores itself too kindly):
  llm        LLM_*        the RAG generator (answers)                    e.g. gpt-6-luna
  eval_llm   EVAL_LLM_*   eval-set builders (eval/build_*.py)            e.g. azure.claude-haiku-5-5
  judge_llm  JUDGE_LLM_*  LLM judges that score the generator's answers  e.g. gemini-3.8-flash
*_BASE_URL / *_API_KEY of the eval and judge endpoints fall back to LLM_BASE_URL / LLM_API_KEY (one proxy can serve
all three).

    from alpha_audit.config import get_settings
    s = get_settings(); s.retrieval.k; s.llm.model; s.eval_llm.model; s.judge_llm.model; s.path("chroma")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


class PathSettings(BaseModel):
    raw_data: Path
    processed: Path
    chroma: Path
    logs: Path


class ChunkingSettings(BaseModel):
    target_max_tokens: int = 350
    header_reserve: int = 64
    overlap: int = 50
    min_section_tokens: int = 50


class RetrievalSettings(BaseModel):
    embedding_model: str
    collection: str
    reranker: str
    k_fetch: int = 50
    bm25_k: int = 0  # BM25 candidates added to the bi-encoder's before reranking (0 = dense only)
    k: int = 5
    per_doc_cap: int | None = None
    not_in_corpus_threshold: float = 0.20


class LLMEndpoint(BaseSettings):
    """One OpenAI-compatible endpoint + model. Non-secret defaults come from config.yaml; the environment overrides."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    rpm: int = 18
    max_attempts: int = 6

    @property
    def env_prefix(self) -> str:
        return self.model_config["env_prefix"]

    @property
    def default_reasoning_effort(self) -> str | None:
        return None


class LLMSettings(LLMEndpoint):
    """The generator: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_ANSWER_REASONING_EFFORT, ..."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=ROOT / ".env", extra="ignore")

    answer_reasoning_effort: str | None = None

    @property
    def default_reasoning_effort(self) -> str | None:
        return self.answer_reasoning_effort


class EvalLLMSettings(LLMEndpoint):
    """Eval-set generation: EVAL_LLM_MODEL, EVAL_LLM_GENERATION_REASONING_EFFORT, ... (URL/key fall back to the
    generator's)."""

    model_config = SettingsConfigDict(env_prefix="EVAL_LLM_", env_file=ROOT / ".env", extra="ignore")

    generation_reasoning_effort: str | None = None  # writing questions / claims
    check_reasoning_effort: str | None = None  # checking the questions / claims it wrote

    @property
    def default_reasoning_effort(self) -> str | None:
        return self.check_reasoning_effort


class JudgeLLMSettings(LLMEndpoint):
    """LLM judges of the generator's answers: JUDGE_LLM_MODEL, JUDGE_LLM_REASONING_EFFORT, ... (URL/key fall back to
    the generator's)."""

    model_config = SettingsConfigDict(env_prefix="JUDGE_LLM_", env_file=ROOT / ".env", extra="ignore")

    reasoning_effort: str | None = None

    @property
    def default_reasoning_effort(self) -> str | None:
        return self.reasoning_effort


FAMILIES = ("claude", "gpt", "gemini", "llama", "mistral", "qwen", "deepseek")


def model_family(model: str | None) -> str | None:
    """'azure.claude-haiku-5-5' -> 'claude', 'gpt-6-luna' -> 'gpt' (provider prefixes ignored)."""
    if not model:
        return None
    name = model.lower().rsplit("/", 1)[-1]
    return next((f for f in FAMILIES if f in name), name)


class Settings(BaseModel):
    paths: PathSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    llm: LLMSettings
    eval_llm: EvalLLMSettings
    judge_llm: JudgeLLMSettings

    def path(self, name: str) -> Path:
        """An absolute path from the `paths` section (config paths are relative to the repo root)."""
        return ROOT / getattr(self.paths, name)

    def check_eval_llm_independent(self, allow_same_family: bool = False) -> None:
        """Refuse to build or judge eval data with the generator's model family (unless explicitly allowed)."""
        gen, ev = model_family(self.llm.model), model_family(self.eval_llm.model)
        if gen and gen == ev and not allow_same_family:
            raise SystemExit(
                f"EVAL_LLM_MODEL ({self.eval_llm.model}) is from the generator's model family ({self.llm.model}). "
                "Use a different family for eval data (same-model bias), or pass --allow-same-family."
            )

    def check_judge_independent(self) -> None:
        """Refuse to judge the generator's answers with its own model family."""
        gen, judge = model_family(self.llm.model), model_family(self.judge_llm.model)
        if not judge:
            raise SystemExit("set JUDGE_LLM_MODEL in .env (see .env.example)")
        if gen == judge:
            raise SystemExit(
                f"JUDGE_LLM_MODEL ({self.judge_llm.model}) is from the generator's model family ({self.llm.model}): "
                "a judge scores its own family's answers too kindly. Use a different family."
            )


def _endpoint(cls: type[LLMEndpoint], yaml_values: dict, **fallback) -> LLMEndpoint:
    """Defaults < config.yaml < environment / .env; unset URL/key then fall back to `fallback`."""
    env = cls()  # only the fields actually set in the environment override config.yaml
    values = {**cls().model_dump(), **yaml_values, **env.model_dump(exclude_unset=True)}
    values.update({k: v for k, v in fallback.items() if values.get(k) is None})
    return cls.model_construct(**values)


@lru_cache
def get_settings(config_path: Path = CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = _endpoint(LLMSettings, raw.get("llm") or {})
    eval_llm = _endpoint(EvalLLMSettings, raw.get("eval_llm") or {}, base_url=llm.base_url, api_key=llm.api_key)
    judge_llm = _endpoint(JudgeLLMSettings, raw.get("judge_llm") or {}, base_url=llm.base_url, api_key=llm.api_key)
    return Settings(paths=raw["paths"], chunking=raw.get("chunking", {}), retrieval=raw["retrieval"],
                    llm=llm, eval_llm=eval_llm, judge_llm=judge_llm)
