from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flag:
    """One guard finding. `guard`: which guard raised it; `kind`: a stable slug for counting in evals."""

    guard: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.guard}:{self.kind}] {self.detail}"
