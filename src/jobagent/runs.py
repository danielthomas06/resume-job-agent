"""Local audit log: every pipeline run is saved with its run_id and trace_id (data/runs/<run_id>.json)."""
import json
from datetime import datetime
from pathlib import Path

RUNS = Path("data/runs")


def save_run(report: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"{report['run_id']}.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    return p


def load_run(run_id: str) -> dict:
    return json.loads((RUNS / f"{run_id}.json").read_text())


def list_runs(limit: int = 30) -> list[dict]:
    """Newest first: run_id, time, candidate, provider/model, trace_id."""
    out = []
    for p in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        r = json.loads(p.read_text())
        out.append({"run_id": r["run_id"], "time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                    "candidate": r["profile"].get("name", ""), "provider": r["meta"]["provider"],
                    "model": r["meta"]["model"], "trace_id": r["trace_id"], "top": r["jobs"][0]["job"]["title"] if r["jobs"] else ""})
    return out
