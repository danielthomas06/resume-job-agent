import asyncio
import tempfile
from pathlib import Path

import httpx
import streamlit as st

from jobagent import feedback
from jobagent.config import get_settings
from jobagent.guardrails import GuardrailError, safe_url, validate_resume_file, validate_resume_text
from jobagent.llm import get_llm
from jobagent.observability import setup_tracing
from jobagent.pipeline.graph import run_pipeline
from jobagent.resume import parse_resume
from jobagent.runs import list_runs, load_run
from jobagent.ui.live import LABELS, Timeline, render


@st.cache_data(ttl=5)
def phoenix_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1.0).status_code < 500
    except httpx.HTTPError:
        return False


st.set_page_config(page_title="Job Agent", page_icon="🧭", layout="wide")
s = get_settings()
setup_tracing(s.otel_exporter_otlp_endpoint, project=s.otel_project, public_url=s.phoenix_public_url)

st.title("🧭 Resume → Jobs agent")
st.caption("Local LLM or Gemini · MCP job sources · embeddings + LLM rerank + grounding critic")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", ["local", "gemini", "auto"],
                            index=["local", "gemini", "auto"].index(s.llm_provider))
    if provider != "local" and not s.gemini_api_key:
        st.warning("GEMINI_API_KEY is not set in .env")
    locations = st.text_input("Locations (first: all sources; rest: Adzuna)", "india")
    st.caption("Redaction of email/phone/links before remote LLM calls is "
               + ("**on**" if s.redact_before_remote else "**off**"))
    phoenix = s.otel_exporter_otlp_endpoint
    if not phoenix:
        st.caption("Tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT is empty)")
    elif phoenix_up(phoenix):
        st.success("Phoenix is running")
        st.markdown(f"[Open traces ↗]({s.phoenix_public_url or phoenix}/projects)")
    else:
        st.error("Phoenix is not reachable, so traces are being dropped")
        st.code("docker compose up -d --wait phoenix", language="bash")
    st.caption(f"Feedback so far: {feedback.counts() or 'none'}")
    st.divider()
    st.subheader("Past runs")
    past = list_runs(15)
    pick = st.selectbox("Reload a saved run", [""] + [r["run_id"] for r in past],
                        format_func=lambda i: "—" if not i else next(
                            f"{r['time'][5:16]} · {r['candidate'] or '?'} · {r['model']}" for r in past if r["run_id"] == i))
    if pick and st.button("Load run"):
        st.session_state["report"] = load_run(pick)

up = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt", "md"])

if st.button("Find jobs", type="primary", disabled=up is None):
    try:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / Path(up.name).name
            path.write_bytes(up.getvalue())
            validate_resume_file(path)
            text = validate_resume_text(parse_resume(path))
    except GuardrailError as e:
        st.error(f"Resume rejected: {e}")
        st.stop()
    locs = [x.strip() for x in locations.split(",") if x.strip()] or None
    with st.status("Running agent…", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        body = st.empty()
        tl = Timeline()

        def on_event(e: dict) -> None:  # called synchronously from the pipeline, in this script thread
            tl.handle(e)
            frac, md = render(tl)
            bar.progress(frac, text=f"{LABELS.get(tl.current, 'Working')}…" if tl.current else "Working…")
            body.markdown(md, unsafe_allow_html=True)

        try:
            st.session_state["report"] = asyncio.run(
                run_pipeline(text, get_llm(provider), locs, s.redact_before_remote, on_event=on_event))
            bar.progress(1.0, text="Done")
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")

rep = st.session_state.get("report")
if rep:
    p = rep["profile"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate", p["name"] or "—")
    c2.metric("Seniority", f"{p['seniority']} · {p['total_years_experience']:.0f}y")
    c3.metric("Jobs considered", rep["pool_size"])
    c4.metric("LLM calls / tokens", f"{rep['usage']['llm_calls']} / "
              f"{rep['usage']['input_tokens'] + rep['usage']['output_tokens']:,}")
    st.caption(f"run `{rep['run_id']}` · trace `{rep['trace_id'][:12]}…` · {rep['meta']['provider']}:{rep['meta']['model']}")
    if rep.get("trace_url"):
        st.markdown(f"[Open this run's trace in Phoenix ↗]({rep['trace_url']})")
    u = rep["usage"]
    if u.get("truncated_calls"):
        st.error(f"{u['truncated_calls']} LLM call(s) hit the context window and may have been truncated. "
                 "Raise OLLAMA_NUM_CTX.")
    with st.expander("Agent timeline: what it did and why", expanded=True):
        if rep.get("plan_rationale"):
            st.info(f"**Search plan rationale:** {rep['plan_rationale']}")
        st.dataframe(rep.get("steps", []), hide_index=True, use_container_width=True,
                     column_config={"node": "Step", "seconds": "Sec", "attempt": "Pass", "note": "What happened"})
    with st.expander(f"LLM requests, tokens and context window ({u['llm_calls']} calls)"):
        peak = u.get("peak_context_utilization")
        a, b, c = st.columns(3)
        a.metric("Prompt tokens", f"{u['input_tokens']:,}")
        b.metric("Completion tokens", f"{u['output_tokens']:,}")
        c.metric("Peak context used", f"{peak:.0%}" if peak is not None else "n/a")
        st.dataframe(u.get("calls", []), hide_index=True, use_container_width=True, column_config={
            "context_utilization": st.column_config.ProgressColumn("ctx used", min_value=0, max_value=1, format="percent")})
    with st.expander("Extracted profile and search plan"):
        st.write("**Searched:**", ", ".join(rep["queries"]))
        st.write("**Skills:**", ", ".join(p["skills"]))
        st.json(p, expanded=False)

    if rep["top_skill_gaps"]:
        st.subheader("Top skill gaps across your best matches")
        st.bar_chart({k: v for k, v in rep["top_skill_gaps"]}, horizontal=True)

    st.subheader("Ranked jobs")
    for i, r in enumerate(rep["jobs"], 1):
        j, jd = r["job"], r["judgement"]
        flagged = any("injection" in f for f in r["flags"])
        label = f"{i}. {j['title']} — {j['company']} · {r['final']:.2f}" + ("  ⚠️" if flagged else "")
        with st.expander(label, expanded=i <= 3):
            st.caption(f"{j['location']} · {j['source']}"
                       + (f" · LLM fit {r['llm_score']}/10" if r["llm_score"] is not None else " · not LLM-judged"))
            if jd:
                st.write(jd["reason"])
                st.write("✅ " + (", ".join(jd["matched_skills"]) or "no grounded matches"))
                st.write("❌ " + (", ".join(jd["missing_skills"]) or "no grounded gaps"))
            for f in r["flags"]:
                st.warning(f"Critic/guardrail: {f}")
            with st.popover("Why this rank?"):
                st.markdown("\n".join(f"{n}. {t}" for n, t in enumerate(r["trail"], 1)))
                if r["features"]:
                    st.json(r["features"], expanded=False)
            if safe_url(j["url"]):
                st.link_button("Open posting", j["url"])
            a, b, _ = st.columns([1, 1, 6])
            if a.button("👍", key=f"up{j['id']}"):
                feedback.record(j["id"], j["title"], j["company"], 1)
                st.toast("Saved")
            if b.button("👎", key=f"dn{j['id']}"):
                feedback.record(j["id"], j["title"], j["company"], -1)
                st.toast("Saved")
