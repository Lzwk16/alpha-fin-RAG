"""LLM calls through the OpenAI-compatible proxy in .env, with structured (pydantic) outputs.

    client = LLMClient()                                   # the generator (LLM_*)
    answer = client.parse(system_prompt, user_prompt, Answer)
    builder = LLMClient(get_settings().eval_llm)           # eval-set builders (EVAL_LLM_*)
    judge = LLMClient(get_settings().judge_llm)            # judges of the generator's answers (JUDGE_LLM_*)

Handles the proxy's request limit (client-side pacing + waiting until the reset time it reports on a 429), omits
`temperature` (reasoning models reject it) and passes `reasoning_effort` when configured. Token usage is tracked in
`client.usage`.
"""

from __future__ import annotations

import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from alpha_audit.config import LLMEndpoint, get_settings

T = TypeVar("T", bound=BaseModel)
RESET_AT_RE = re.compile(r"resets at: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC")


class RateLimiter:
    """At most `rpm` request starts in any `window` seconds (blocking)."""

    def __init__(self, rpm: int, window: float = 60.0):
        self.rpm, self.window, self.starts = rpm, window, deque()

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            while self.starts and now - self.starts[0] >= self.window:
                self.starts.popleft()
            if len(self.starts) < self.rpm:
                self.starts.append(now)
                return
            time.sleep(self.window - (now - self.starts[0]) + 0.1)


def seconds_until_reset(err: Exception) -> float:
    """How long to wait after a 429: until the reset time the proxy reports, else 60 s."""
    m = RESET_AT_RE.search(str(err))
    if not m:
        return 60.0
    reset = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return max(1.0, (reset - datetime.now(UTC)).total_seconds() + 2)


class LLMClient:
    def __init__(
        self,
        settings: LLMEndpoint | None = None,
        reasoning_effort: str | None = "config",
    ):
        """`settings`: get_settings().llm (default) or .eval_llm. reasoning_effort="config" uses that endpoint's
        default (generator: answer_reasoning_effort; eval: check_reasoning_effort); None never sends it."""

        self.settings = s = settings or get_settings().llm
        missing = [
            name for name in ("base_url", "api_key", "model") if not getattr(s, name)
        ]
        if missing:
            raise RuntimeError(
                f"set {', '.join(s.env_prefix + m.upper() for m in missing)} in .env (see .env.example)"
            )
        self.client = OpenAI(
            base_url=s.base_url, api_key=s.api_key.get_secret_value(), max_retries=0
        )
        self.model = s.model
        self.reasoning_effort = (
            s.default_reasoning_effort
            if reasoning_effort == "config"
            else reasoning_effort
        )
        self.limiter = RateLimiter(s.rpm)
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def parse(self, system: str, user: str, schema: type[T]) -> T:
        """One chat completion whose answer must fit `schema`. Retries on rate limits / transient errors and re-asks
        once if the answer cannot be parsed."""

        extra = (
            {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        reasked = False
        for attempt in range(self.settings.max_attempts):
            self.limiter.wait()
            bad_answer = None
            try:
                resp = self.client.beta.chat.completions.parse(
                    model=self.model, messages=messages, response_format=schema, **extra
                )
            except RateLimitError as e:
                time.sleep(seconds_until_reset(e))
                continue
            except (APIConnectionError, APITimeoutError, InternalServerError):
                time.sleep(min(60, 2**attempt))
                continue
            except (
                ValidationError,
                ValueError,
            ) as e:  # the SDK raises when the JSON does not fit the schema
                self.usage["calls"] += 1
                bad_answer = f"(unparsable: {type(e).__name__})"
            else:
                self.usage["calls"] += 1
                if resp.usage:
                    self.usage["input_tokens"] += resp.usage.prompt_tokens
                    self.usage["output_tokens"] += resp.usage.completion_tokens
                msg = resp.choices[0].message
                if msg.parsed is not None:
                    return msg.parsed
                bad_answer = msg.content or msg.refusal or ""
            if reasked:
                break
            reasked = True  # one repair re-ask: tell the model its answer did not fit the schema
            messages = [
                *messages,
                {"role": "assistant", "content": bad_answer},
                {
                    "role": "user",
                    "content": "That answer did not match the required JSON schema. "
                    "Reply again with JSON that matches it exactly.",
                },
            ]
        raise RuntimeError(
            f"no parsable answer from {self.model} after {self.settings.max_attempts} attempts"
        )
