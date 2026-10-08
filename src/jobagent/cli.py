import asyncio
import json
from pathlib import Path

import typer
from rich import print_json
from rich.console import Console

from jobagent.config import get_settings
from jobagent.guardrails import GuardrailError, validate_resume_file, validate_resume_text
from jobagent.llm import get_llm
from jobagent.observability import setup_tracing
from jobagent.resume import extract_profile, parse_resume

app = typer.Typer(help="jobagent: resume -> jobs agent")
console = Console()


def _load_resume(path: Path) -> str:
    try:
        validate_resume_file(path)
        return validate_resume_text(parse_resume(path))
    except GuardrailError as e:
        console.print(f"[red]Rejected resume:[/red] {e}")
        raise typer.Exit(2)


@app.callback()
def main():
    """jobagent CLI."""


@app.command()
def profile(
    resume: Path = typer.Argument(..., exists=True, help="PDF/DOCX/TXT resume"),
    provider: str = typer.Option(None, help="local | gemini | auto (default from .env)"),
):
    """Extract a structured profile from a resume."""
    s = get_settings()
    setup_tracing(s.otel_exporter_otlp_endpoint, project=s.otel_project, public_url=s.phoenix_public_url)
    llm = get_llm(provider)
    console.print(f"[dim]provider={llm.provider} model={llm.model}[/dim]")
    p = extract_profile(_load_resume(resume), llm, s.redact_before_remote)
    print_json(p.model_dump_json())


@app.command()
def jobs(
    query: str = typer.Argument(..., help="e.g. 'machine learning engineer'"),
    location: str = typer.Option("", help="country/city/remote"),
    limit: int = typer.Option(5),
):
    """Search jobs through the jobs-mcp server."""
    from jobagent.mcp_client import jobs_client

    setup_tracing(get_settings().otel_exporter_otlp_endpoint, project=get_settings().otel_project,
                  public_url=get_settings().phoenix_public_url)

    async def run():
        async with jobs_client() as c:
            console.print(f"[dim]sources: {await c.call('list_sources')}[/dim]")
            return await c.call("search_jobs", query=query, location=location, limit=limit)

    for j in asyncio.run(run()):
        console.print(f"[bold]{j['title']}[/bold] @ {j['company']} ({j['location']}) "
                      f"[dim]{j['source']}[/dim]\n  {j['url']}")


@app.command()
def run(
    resume: Path = typer.Argument(..., exists=True),
    provider: str = typer.Option(None, help="local | gemini | auto"),
    location: list[str] = typer.Option(None, help="repeatable; first is used for all sources, the rest for Adzuna only (default: india)"),
    json_out: Path = typer.Option(None, help="also write the full report as JSON"),
    explain: bool = typer.Option(False, help="print each job's score trail (why it ranked here)"),
):
    """Full pipeline: resume -> profile -> search -> rank -> report."""
    from rich.markup import escape

    from jobagent.pipeline.graph import run_pipeline

    s = get_settings()
    setup_tracing(s.otel_exporter_otlp_endpoint, project=s.otel_project, public_url=s.phoenix_public_url)
    llm = get_llm(provider)
    console.print(f"[dim]provider={llm.provider} model={llm.model}[/dim]")
    def live(e: dict) -> None:
        if e["type"] == "node_end":
            console.print(f"[green]✓[/green] {e['step']['node']} [dim]{e['step']['seconds']:.1f}s[/dim]")
        elif e["type"] == "job_judged":
            console.print(f"  [dim]judged {e['i']}/{e['total']}: {escape(e['title'])} -> {e['llm_score']}/10[/dim]")

    rep = asyncio.run(run_pipeline(_load_resume(resume), llm, location or None, s.redact_before_remote, on_event=live))
    console.print(f"[dim]run={rep['run_id']} queries={rep['queries']} pool={rep['pool_size']}[/dim]")
    if rep["trace_url"]:
        console.print(f"[dim]trace: {rep['trace_url']}[/dim]\n")
    for i, r in enumerate(rep["jobs"], 1):
        j, jd = r["job"], r["judgement"]
        llm_s = f"llm {r['llm_score']}/10" if r["llm_score"] is not None else "not judged"
        console.print(f"[bold]{i}. {j['title']}[/bold] @ {j['company']} ({j['location']}) "
                      f"[cyan]{r['final']:.2f}[/cyan] [dim]{llm_s}, {j['source']}[/dim]\n   {j['url']}")
        if jd:
            console.print(f"   {jd['reason']}\n   [green]match:[/green] {', '.join(jd['matched_skills'])}"
                          f"  [red]gaps:[/red] {', '.join(jd['missing_skills'])}")
        if r["flags"]:
            console.print(f"   [yellow]critic: {'; '.join(r['flags'][:3])}[/yellow]")
        if explain:
            console.print("   [dim]" + "\n   ".join(escape(t) for t in r["trail"]) + "[/dim]")
    if explain:
        console.print("\n[bold]Agent timeline[/bold]")
        for st_ in rep["steps"]:
            console.print(f"  {st_['node']:<22} {st_['seconds']:>6.1f}s  [dim]{escape(st_['note'])}[/dim]")
        console.print("\n[bold]LLM requests[/bold] (prompt/completion tokens, context window use)")
        for c in rep["usage"]["calls"]:
            util = f"{c['context_utilization']:.0%}" if c["context_utilization"] is not None else "n/a"
            console.print(f"  #{c['n']:<3} {c['purpose'] or '-':<16} try {c['attempt']}  "
                          f"{c['prompt_tokens']:>5}+{c['completion_tokens']:<4} tok  ctx {util:>4} of {c['context_window']}  "
                          f"{c['latency_s']:>5.1f}s  {c['finish_reason']}{'  [red]TRUNCATED[/red]' if c['truncated'] else ''}")
    if rep["top_skill_gaps"]:
        console.print("\n[bold]Top skill gaps:[/bold] " + ", ".join(f"{k} ({v})" for k, v in rep["top_skill_gaps"]))
    if json_out:
        json_out.write_text(json.dumps(rep, indent=2, default=str))


@app.command(name="eval")
def eval_cmd(
    provider: str = typer.Option(None, help="local | gemini | auto"),
    persona: list[str] = typer.Option(None, help="restrict to golden persona(s)"),
    no_llm_rank: bool = typer.Option(False, help="skip LLM rerank (embedding/prelim only)"),
    pool: str = typer.Option("all", help="all | core (30 easy jobs) | hard (15 near-misses only)"),
    judge: bool = typer.Option(False, help="LLM-judge the explanations for faithfulness"),
    judge_provider: str = typer.Option(None, help="judge provider (default: same as --provider)"),
    judge_model: str = typer.Option(None, help="judge model, e.g. qwen2.5:14b-instruct"),
):
    """Run the offline eval suite and save a JSON report under evals/results/."""
    from rich.markup import escape
    from rich.table import Table

    from jobagent.evals.runner import run_eval, save

    s = get_settings()
    setup_tracing(s.otel_exporter_otlp_endpoint, project="jobagent-evals", public_url=s.phoenix_public_url)
    llm = get_llm(provider)
    console.print(f"[dim]evaluating provider={llm.provider} model={llm.model}[/dim]")
    judge_llm = get_llm(judge_provider or provider, judge_model) if judge else None
    if judge_llm:
        console.print(f"[dim]judge={judge_llm.provider}:{judge_llm.model}[/dim]")
    rep = run_eval(llm, persona or None, llm_rank=not no_llm_rank, pool=pool, judge=judge_llm)
    t = Table("metric", "value")
    for k, v in rep["summary"].items():
        t.add_row(escape(k), f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(t)
    console.print(f"usage: {({k: v for k, v in rep['usage'].items() if k != 'calls'})}\nsaved: {save(rep)}\ntrace: {rep['trace_url']}")


@app.command()
def runs(show: str = typer.Argument(None, help="run id to print as JSON; omit to list recent runs")):
    """List saved runs (data/runs) with their trace ids, or show one."""
    from rich.table import Table

    from jobagent.runs import list_runs, load_run

    if show:
        print_json(json.dumps(load_run(show), default=str))
        return
    t = Table("run_id", "time", "candidate", "model", "top result", "trace_id")
    for r in list_runs():
        t.add_row(r["run_id"], r["time"], r["candidate"], f"{r['provider']}:{r['model']}", r["top"], r["trace_id"][:12] + "…")
    console.print(t)


@app.command()
def ui():
    """Launch the Streamlit UI."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "ui" / "app.py"
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))


if __name__ == "__main__":
    app()
