"""
analyze_narrative_debug.py

Reads logs/narrative_debug.jsonl (written by
app/reporting/narrative_debug_log.py during a real pipeline run) and
prints, per entry: ticker, prompt size, raw output size, which
headings matched, whether Executive Summary failed, and -- on any
failure -- the actual raw model output so the failure mode can be
inspected directly instead of guessed at.
"""

import json
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "narrative_debug.jsonl"


def main():
    if not LOG_PATH.exists():
        print(f"No log at {LOG_PATH}")
        return

    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Total narrative generation calls logged: {len(entries)}\n")

    header = f"{'Ticker':<7}{'PromptChars':>12}{'PromptWords':>12}{'OutputChars':>12}{'HeadingsMatched':>17}{'ExecFail':>10}"
    print(header)
    print("-" * len(header))
    for e in entries:
        headings = len(e["headings_matched"])
        exec_fail = (not e["executive_summary_heading_matched"]) or e["executive_summary_content_empty"]
        print(f"{e['ticker']:<7}{e['prompt_char_count']:>12}{e['prompt_word_count']:>12}"
              f"{e['raw_output_char_count']:>12}{headings:>17}{str(exec_fail):>10}")

    failures = [e for e in entries if (not e["executive_summary_heading_matched"]) or e["executive_summary_content_empty"]]
    print(f"\nExecutive Summary failures: {len(failures)}/{len(entries)}")

    for e in failures:
        print(f"\n{'=' * 70}\nFAILURE: {e['ticker']}  (heading_matched={e['executive_summary_heading_matched']}, "
              f"content_empty={e['executive_summary_content_empty']})")
        print(f"Prompt: {e['prompt_char_count']} chars / {e['prompt_word_count']} words")
        print(f"Headings actually matched in raw output: {e['headings_matched']}")
        print(f"\n--- RAW MODEL OUTPUT (first 2000 chars) ---")
        print(e["raw_output"][:2000])
        print(f"--- end raw output snippet ---")

    print(f"\n{'=' * 70}")
    print("Prompt size across all calls (for correlating with grounding trend):")
    for e in entries:
        print(f"  {e['ticker']:<7} prompt={e['prompt_char_count']:>6} chars  output={e['raw_output_char_count']:>6} chars")


if __name__ == "__main__":
    main()
