"""
narrative_debug_log.py

Diagnostic-only logging for narrative_builder.py -- added specifically
to investigate the recurring "Not available for this report" failure
on Executive Summary. Appends one JSON line per report generation
call to logs/narrative_debug.jsonl, capturing the full prompt, the
full raw LLM output, and which section headings _split_sections()
actually matched BEFORE the "Not available" placeholder is applied --
so a failure can be diagnosed from the real raw model output instead
of guessed at. Also captures prompt size (char/word count) for
correlating against grounding score.

Not wired into any user-facing behavior -- purely diagnostic. Not a
fix; per the standing instruction, this exists so a fix attempt (if
any) is based on an inspected failure mode, not a guess.
"""

import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join("logs", "narrative_debug.jsonl")


def log_narrative_call(ticker, prompt, raw_output, raw_sections, recommendation_rating):
    """
    raw_sections: the dict returned directly by _split_sections(raw),
    BEFORE build_narrative_sections() fills in "Not available for
    this report." placeholders for anything unmatched -- this is what
    lets the log distinguish "heading never matched at all" from
    "heading matched but content came out empty/whitespace."
    """
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

    exec_summary = raw_sections.get("Executive Summary")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "recommendation_rating": recommendation_rating,
        "prompt_char_count": len(prompt),
        "prompt_word_count": len(prompt.split()),
        "raw_output_char_count": len(raw_output),
        "raw_output_word_count": len(raw_output.split()),
        "raw_output": raw_output,
        "prompt": prompt,
        "headings_matched": list(raw_sections.keys()),
        "executive_summary_heading_matched": "Executive Summary" in raw_sections,
        "executive_summary_content_empty": exec_summary is not None and not exec_summary.strip(),
        "executive_summary_text": exec_summary,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
