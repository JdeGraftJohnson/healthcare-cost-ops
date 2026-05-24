"""Champion-challenger comparison from the JSONL run ledger.

Reads `logs/forecast_runs__<pipeline>.jsonl`, treats the previous successful
run for the same pipeline as the champion, and produces a markdown delta vs
the latest (challenger) run. Designed to slot into CI: exit non-zero when
the challenger regresses by more than `--tol` on any tracked metric.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("forecast.compare")


PRIMARY_METRICS = (
    "final_test__sarima__mape",
    "final_test__sarima__mase",
    "final_test__sarima__coverage_80",
    "final_test__ensemble__mape",
    "final_test__ensemble__coverage_80",
    "backtest__sarima__mape",
    "backtest__sarima__mase",
)


@dataclass
class Delta:
    metric: str
    champion: float | None
    challenger: float | None
    delta: float | None
    pct_change: float | None
    direction: str   # 'better' | 'worse' | 'same' | 'n/a'


def _read_ledger(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pick_champion_challenger(records: list[dict]) -> tuple[dict | None, dict | None]:
    """Most-recent record is the challenger; the one before it is the champion."""
    completed = [r for r in records if r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("ended_at", 0))
    if not completed:
        return None, None
    if len(completed) == 1:
        return None, completed[-1]
    return completed[-2], completed[-1]


def compute_deltas(
    champion: dict | None, challenger: dict,
    metric_keys: tuple[str, ...] = PRIMARY_METRICS,
) -> list[Delta]:
    champ_metrics = (champion or {}).get("metrics", {})
    chal_metrics = challenger.get("metrics", {})
    out: list[Delta] = []
    for k in metric_keys:
        c = champ_metrics.get(k)
        x = chal_metrics.get(k)
        if x is None:
            continue
        if c is None:
            out.append(Delta(k, None, x, None, None, "n/a"))
            continue
        delta = x - c
        pct = (delta / c) if abs(c) > 1e-9 else None
        # For "lower is better" metrics (mape / mase / rmse / mae / bias / pinball),
        # negative delta is "better". For coverage and directional_acc, positive
        # delta is "better".
        higher_is_better = any(
            k.endswith(suffix) for suffix in (
                "__coverage_80", "__coverage_95", "__r2", "__directional_acc",
            )
        )
        if abs(delta) < 1e-9:
            direction = "same"
        elif (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
            direction = "better"
        else:
            direction = "worse"
        out.append(Delta(k, c, x, delta, pct, direction))
    return out


def render_md(
    champion: dict | None, challenger: dict, deltas: list[Delta],
) -> str:
    lines = ["# Forecast pipeline — champion vs challenger\n"]
    lines.append(f"- challenger: `{challenger.get('run_id', 'unknown')}` "
                 f"({challenger.get('name', '?')}, ended {challenger.get('ended_at', '?')})")
    if champion:
        lines.append(f"- champion:   `{champion.get('run_id', 'unknown')}` "
                     f"({champion.get('name', '?')}, ended {champion.get('ended_at', '?')})")
    else:
        lines.append("- champion:   (none — first recorded run)")
    lines.append("")
    if not deltas:
        lines.append("_No tracked metrics present in this run._")
        return "\n".join(lines)
    lines.append("| metric | champion | challenger | Δ | % | direction |")
    lines.append("|---|---|---|---|---|---|")
    for d in deltas:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "—"
        pct = f"{d.pct_change*100:+.2f}%" if d.pct_change is not None else "—"
        delta = fmt(d.delta) if d.delta is not None else "—"
        emoji = {"better": "[OK]", "worse": "[REGRESSION]",
                 "same": "[same]", "n/a": "[new]"}[d.direction]
        lines.append(
            f"| `{d.metric}` | {fmt(d.champion)} | {fmt(d.challenger)} "
            f"| {delta} | {pct} | {emoji} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="forecast-compare",
        description="Champion vs challenger from forecast run ledger.",
    )
    p.add_argument("--ledger", required=True, help="path to logs/forecast_runs__<name>.jsonl")
    p.add_argument("--out", default="-", help="output path or '-' for stdout")
    p.add_argument("--tol", type=float, default=0.05,
                   help="fractional regression tolerance; >tol on any metric exits non-zero")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    records = _read_ledger(a.ledger)
    if not records:
        print(f"[FAIL] no records in {a.ledger}", file=sys.stderr)
        return 2
    champion, challenger = pick_champion_challenger(records)
    if challenger is None:
        print("[FAIL] no completed challenger run", file=sys.stderr)
        return 2
    deltas = compute_deltas(champion, challenger)
    md = render_md(champion, challenger, deltas)
    if a.out == "-":
        print(md)
    else:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        print(f"[OK] wrote {a.out}")
    # CI exit: fail if any "worse" exceeds tol.
    worst = max(
        ((abs(d.pct_change) if d.pct_change is not None else 0.0)
         for d in deltas if d.direction == "worse"),
        default=0.0,
    )
    if worst > a.tol:
        print(f"[REGRESSION] worst pct_change={worst:.4f} > tol={a.tol}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
