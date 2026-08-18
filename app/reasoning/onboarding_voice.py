"""
onboarding_voice.py

Maps a free-text SPOKEN answer onto exactly one of a fixed onboarding
field's known options -- e.g. "I'd say somewhere in the middle" ->
"Moderate". Same controlled-mapping philosophy as chat_router.py's
classify_intent(): a single short HostedProvider call constrained to a
known label set, parsed with the same regex-based structured-output
pattern, never a free-form LLM guess written straight to the database.
GET /v1/auth/onboarding's own field_validator (app/api/main.py) is the
final safety net regardless -- this module only narrows a spoken
sentence down to a candidate value, it never bypasses that validation.

The five field option-lists mirror the fixed tuples app/api/main.py's
OnboardingRequest validates against (_RISK_TOLERANCE_LEVELS etc) and
OnboardingForm.tsx's own RISK_LEVELS/GOALS/HORIZONS constants -- kept
independent rather than imported, the same duplication already
accepted between the frontend and backend copies of these same lists
(a backend reasoning module importing from the API layer would invert
the dependency direction main.py already establishes: main.py imports
FROM reasoning modules, never the other way around).
"""

import re
from typing import Optional

from app.core.llm_provider import HostedProvider, LLMProviderError

FIELD_OPTIONS: dict[str, tuple[str, ...]] = {
    "risk_tolerance": ("Conservative", "Moderate", "Aggressive"),
    "investment_goal": ("Wealth Growth", "Retirement", "Income", "Capital Preservation"),
    "investment_horizon": ("Short-term (<3y)", "Medium (3-7y)", "Long-term (7y+)"),
    "interested_in_crypto": ("Yes", "No"),
    "interested_in_real_estate": ("Yes", "No"),
}

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)

# allam-2-7b, not a gpt-oss/reasoning model: same reasoning as
# chat_router.py's CHAT_MODEL -- no hidden chain-of-thought tax, so it
# reliably fits this module's tight 20-token budget. Was previously
# pinned to llama-3.3-70b-versatile, which Groq has fully removed.
_MODEL = "allam-2-7b"


def _build_prompt(options: tuple[str, ...], answer_text: str) -> str:
    option_list = ", ".join(options)
    return f"""A user was asked a multiple-choice question during onboarding and gave this spoken answer. Map it to EXACTLY ONE of these options: {option_list}

If the answer clearly doesn't match any option, or is too ambiguous to tell, respond with NONE -- do not guess.

SPOKEN ANSWER: {answer_text}

Respond in EXACTLY this format, nothing else:
ANSWER: <one of the options above, or NONE>"""


def classify_onboarding_answer(field: str, answer_text: str) -> Optional[str]:
    """Returns the canonical option string for `field` (exact spelling
    from FIELD_OPTIONS, never the LLM's own casing/punctuation) or None
    if the answer didn't clearly map to one. Raises KeyError for an
    unknown `field` -- that's a caller bug (a field name typo), not a
    degrade-gracefully case like an ambiguous spoken answer is."""
    options = FIELD_OPTIONS[field]

    try:
        provider = HostedProvider(model=_MODEL)
        raw = provider.generate(_build_prompt(options, answer_text), max_new_tokens=20)
    except LLMProviderError:
        return None

    match = _ANSWER_RE.search(raw)
    if not match:
        return None
    candidate = match.group(1).strip()

    for option in options:
        if option.lower() == candidate.lower():
            return option
    return None
