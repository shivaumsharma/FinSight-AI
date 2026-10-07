"""
delisted_rerun.py -- RESEARCH SANDBOX. Unattended finisher for the G1 survivorship fix.

Waits until the Tiingo downloader (research/tiingo_prices.py run) has exited AND logged "finished", then runs the whole
rerun in order, writing everything to research/data/final_rerun.log and a compact before/after to
research/data/final_rerun_summary.txt:

    delisted_match -> delisted_build -> pit_composite run_new -> merge scores -> pit_composite_ic -> factor_ic ->
    ml_cross_section -> ml_cross_section_4q

Safe to re-run; every step is idempotent (originals are kept as *_pre_delisted.* / p2b_scores_survivors.csv).
Run: python research/delisted_rerun.py [--now]     (--now skips the wait)
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research" / "data"
LOG = DATA / "final_rerun.log"
SUMMARY = DATA / "final_rerun_summary.txt"


def downloader_running():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*tiingo_prices.py run*' "
                          "-and $_.Name -eq 'python.exe' } | Measure-Object).Count"], capture_output=True, text=True).stdout
    return int((out.strip() or "0")) > 0


def run(cmd, note):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {note}: {' '.join(cmd)} =====\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"----- exit code {r.returncode}\n")
    return r.returncode


def merge_scores():
    base = pd.read_csv(DATA / "p2b_scores_survivors.csv")
    new = pd.read_csv(DATA / "p2b_scores_delisted.csv")
    pd.concat([base, new[~new.ticker.isin(base.ticker)]]).to_csv(DATA / "p2b_scores.csv", index=False)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"merged: {new.ticker.nunique()} former members into p2b_scores.csv\n")


def tail_of(cmd_log_marker, n=25):
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    block = text.split(cmd_log_marker)[-1]
    return "\n".join(block.strip().splitlines()[-n:])


def main():
    if "--now" not in sys.argv:
        while True:
            done = "finished:" in (DATA / "tiingo_run.log").read_text(encoding="utf-8", errors="ignore")
            if done and not downloader_running():
                break
            time.sleep(120)
    LOG.write_text("", encoding="utf-8")
    py = sys.executable
    steps = [
        ([py, "research/delisted_match.py"], "match"),
        ([py, "research/delisted_build.py"], "build"),
        ([py, "research/pit_composite.py", "run_new"], "score new"),
    ]
    for cmd, note in steps:
        if run(cmd, note) != 0:
            SUMMARY.write_text(f"STOPPED at step '{note}'. See final_rerun.log\n")
            return
    merge_scores()
    for cmd, note in [([py, "research/pit_composite_ic.py"], "composite IC"), ([py, "research/factor_ic.py"], "factors"),
                      ([py, "research/ml_cross_section.py"], "learned model 1q"), ([py, "research/ml_cross_section_4q.py"], "learned model 4q")]:
        run(cmd, note)
    m = pd.read_csv(DATA / "delisted_map.csv")
    lines = ["FINAL SURVIVORSHIP-CORRECTED RERUN", f"matching: {m.status.value_counts().to_dict()}",
             f"REVIEW tickers (not used, need a human): {m[m.status == 'REVIEW'].ticker.tolist()}",
             f"NOMATCH: {m[m.status == 'NOMATCH'].ticker.tolist()}", "",
             "--- composite IC ---", tail_of("composite IC", 22), "",
             "--- factors ---", tail_of("factors", 40)[:2500], "",
             "--- learned model 4q ---", tail_of("learned model 4q", 9)]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
