from alpha_audit.guards.flags import Flag
from alpha_audit.guards.grounding import check_grounding
from alpha_audit.guards.numeric import check_numbers

# Logged with each eval run, because guard changes alter scores without changing the prompt. Bump when they change.
# g2: quotes may come from an excerpt's heading (2026-11-07)
GUARD_VERSION = "g2"

__all__ = ["GUARD_VERSION", "Flag", "check_grounding", "check_numbers"]
