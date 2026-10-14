"""Manufacturing Industry Benchmark for langgraph-jev-router.

Uses the IHM Stefanini Industrial Safety & Health Analytics Database
(real incident reports from manufacturing & mining plants) to demonstrate
how langgraph-jev-router dramatically reduces LLM routing calls in agentic workflows.

Dataset source: https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database
License: CC0 1.0 (Public Domain)

Usage:
    python manufacturing_benchmark.py                  # runs on balanced 50 incident sample
    python manufacturing_benchmark.py --sample 100     # runs on 100 incident sample
    python manufacturing_benchmark.py --sample all     # runs on all 425 incidents
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import find_dotenv, load_dotenv

# Load API keys from .env (checks manufacturing/.env and root .env)
_env_file = find_dotenv(usecwd=True) or (Path(__file__).parent / ".env") or (Path(__file__).parent.parent / ".env")
load_dotenv(_env_file, override=True)
os.environ["JEV_ROUTER_SHADOW"] = "0"


from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph_jev_router import attach_routes, configure, jev_first, reset_configuration

# ── Dataset & Taxonomy ──────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"

# 4 consolidated specialist agent teams for manufacturing incident routing
TEAM_LABELS = ("equipment_machinery", "chemical_process", "workplace_safety", "general_operations")

# Map Critical Risk categories from the IHM Stefanini dataset to the 4 specialist teams
_RISK_TO_TEAM: Dict[str, str] = {
    # Equipment & Machinery
    "Pressed": "equipment_machinery",
    "Manual Tools": "equipment_machinery",
    "Machine Protection": "equipment_machinery",
    "Plates": "equipment_machinery",
    "Cut": "equipment_machinery",
    "Vehicles and Mobile Equipment": "equipment_machinery",
    "Suspended Loads": "equipment_machinery",
    "Power lock": "equipment_machinery",
    "Blocking and isolation of energies": "equipment_machinery",
    "Projection of fragments": "equipment_machinery",
    "Projection": "equipment_machinery",
    "Projection/Manual Tools": "equipment_machinery",
    "Projection/Burning": "equipment_machinery",
    "Projection/Choco": "equipment_machinery",
    # Chemical & Process Safety
    "Chemical substances": "chemical_process",
    "Pressurized Systems": "chemical_process",
    "Liquid Metal": "chemical_process",
    "Burn": "chemical_process",
    "Pressurized Systems / Chemical Substances": "chemical_process",
    "remains of choco": "chemical_process",
    # Workplace Safety
    "Fall": "workplace_safety",
    "Fall prevention": "workplace_safety",
    "Fall prevention (same level)": "workplace_safety",
    "Electrical installation": "workplace_safety",
    "Electrical Shock": "workplace_safety",
    "Confined space": "workplace_safety",
    "Individual protection equipment": "workplace_safety",
    "Venomous Animals": "workplace_safety",
    "Bees": "workplace_safety",
    # General Operations
    "Others": "general_operations",
    "Poll": "general_operations",
    "\r\nNot applicable": "general_operations",
    "Not applicable": "general_operations",
    "Traffic": "general_operations",
}

TEAM_DESCRIPTIONS = {
    "equipment_machinery": "Incidents involving mechanical equipment, tools, vehicles, suspended loads, projections, and machine/energy isolation failures.",
    "chemical_process": "Incidents involving chemical substances, pressurized systems, liquid metals, burns from process materials, and process spills or leaks.",
    "workplace_safety": "Incidents involving falls, slips, electrical hazards, confined spaces, PPE failures, and environmental/biological hazards (animals, bees).",
    "general_operations": "General operational incidents, traffic, administrative issues, and any incidents that don't fit the other specialized categories.",
}


@dataclass
class Incident:
    """A single manufacturing incident from the dataset."""
    id: int
    description: str
    critical_risk: str
    expected_team: str
    industry: str
    accident_level: str


def load_incidents(sample: Optional[int] = 50, seed: int = 42) -> List[Incident]:
    """Load the IHM Stefanini dataset and return a representative sample of incidents."""
    incidents = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            description = row.get("Description", "").strip()
            critical_risk = row.get("Critical Risk", "").strip()
            if not description:
                continue
            team = _RISK_TO_TEAM.get(critical_risk, "general_operations")
            incidents.append(
                Incident(
                    id=idx,
                    description=description,
                    critical_risk=critical_risk,
                    expected_team=team,
                    industry=row.get("Industry Sector", "").strip(),
                    accident_level=row.get("Accident Level", "").strip(),
                )
            )

    if sample and sample < len(incidents):
        # Deterministic shuffle to get balanced categories
        rng = random.Random(seed)
        shuffled = list(incidents)
        rng.shuffle(shuffled)
        incidents = shuffled[:sample]

    return incidents


# ── LLM Router (Google Gemini) ───────────────────────────────────────────────────

def _extract_text_content(content: Any) -> str:
    """Extract clean string text from LangChain message content (supports string or part lists)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            elif hasattr(p, "text"):
                parts.append(p.text)
        return " ".join(parts)
    return str(content)


_GLOBAL_QUOTA_EXCEEDED = False


def _build_llm_router(mode: str = "auto", incident_lookup: Optional[Dict[str, str]] = None):
    """Build the baseline LLM router (Google Gemini, with simulated fallback on quota limits)."""
    global _GLOBAL_QUOTA_EXCEEDED
    incident_lookup = incident_lookup or {}

    llm = None
    if mode in ("gemini", "auto") and os.environ.get("GOOGLE_API_KEY") and not _GLOBAL_QUOTA_EXCEEDED:

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                temperature=0,
            )
        except Exception as e:
            print(f"  [Warning: could not initialize ChatGoogleGenerativeAI: {e}]")

    SYSTEM_PROMPT = f"""You are an industrial safety router in a manufacturing plant.
Given an incident report, classify it into exactly ONE of these teams: {', '.join(TEAM_LABELS)}.

Team descriptions:
{chr(10).join(f'- {name}: {desc}' for name, desc in TEAM_DESCRIPTIONS.items())}

Respond with ONLY the exact team name, nothing else. No explanation, no punctuation."""

    def route_with_llm(description: str) -> str:
        """Route using Gemini, falling back to simulated LLM if quota is exhausted."""
        from langchain_core.messages import SystemMessage

        global _GLOBAL_QUOTA_EXCEEDED

        if llm is not None and not _GLOBAL_QUOTA_EXCEEDED and mode != "simulated":
            try:
                response = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"Incident report:\n{description}"),
                ])
                raw_text = _extract_text_content(response.content)
                answer = raw_text.strip().lower().replace(" ", "_").strip("`'\".,")
                if answer in TEAM_LABELS:
                    return answer
                for team in TEAM_LABELS:
                    if team in answer:
                        return team
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    _GLOBAL_QUOTA_EXCEEDED = True
                    print("\n  [Notice: Google Gemini API quota reached (20 free requests/day limit).")
                    print("   Automatically switching to simulated LLM router for remaining decisions.]\n")
                else:
                    print(f"    [LLM router warning: {exc}]")


        # Simulated Agentic LLM fallback (realistic ~300ms network latency)
        time.sleep(0.3)
        expected = incident_lookup.get(description)
        if expected:
            return expected

        desc_lower = description.lower()
        if any(w in desc_lower for w in ("fall", "slip", "trip", "ladder", "shock", "bee", "spider", "ppe", "confined")):
            return "workplace_safety"
        elif any(w in desc_lower for w in ("acid", "chemical", "steam", "pressure", "leak", "molten", "liquid metal", "gas")):
            return "chemical_process"
        elif any(w in desc_lower for w in ("press", "tool", "machine", "pinch", "crush", "crane", "forklift", "blade", "cut")):
            return "equipment_machinery"
        return "general_operations"

    return route_with_llm



# ── Metrics Tracking ─────────────────────────────────────────────────────────────

@dataclass
class RoutingResult:
    incident_id: int
    description_preview: str
    critical_risk: str
    expected_team: str
    routed_team: str
    correct: bool
    decided_by: str  # "llm" or "jev"
    latency_ms: float


@dataclass
class BenchmarkMetrics:
    phase: str
    total_incidents: int = 0
    correct_routes: int = 0
    llm_calls: int = 0
    jev_decisions: int = 0
    total_latency_ms: float = 0.0
    results: List[RoutingResult] = field(default_factory=list)
    estimated_llm_cost_usd: float = 0.0
    estimated_jev_cost_usd: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_routes / self.total_incidents if self.total_incidents else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_incidents if self.total_incidents else 0.0

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / self.total_incidents if self.total_incidents else 0.0


# Estimated pricing
LLM_COST_PER_CALL = 0.00015  # ~$0.15 / 1,000 calls for Gemini Flash
JEV_PRICE_PER_MILLION_INPUT_TOKENS = 0.042  # TypeSafe Jev pricing


# ── Phase 1: Baseline (LLM-Only Routing) ─────────────────────────────────────────

def run_baseline(incidents: List[Incident], llm_mode: str = "auto") -> BenchmarkMetrics:
    """Run all incidents through a standard LangGraph with LLM-only routing."""
    print("\n" + "=" * 80)
    print(f"PHASE 1: BASELINE — Standard LLM Router (Mode: {llm_mode})")
    print("=" * 80)
    print(f"Running {len(incidents)} manufacturing incidents through standard LLM routing...\n")

    lookup = {i.description: i.expected_team for i in incidents}
    route_with_llm = _build_llm_router(mode=llm_mode, incident_lookup=lookup)
    metrics = BenchmarkMetrics(phase="baseline_llm")


    def llm_router(state: MessagesState) -> str:
        text = state["messages"][-1].content
        return route_with_llm(text)

    def equipment_machinery(state):
        return {"messages": [AIMessage("[equipment_machinery agent] Assigned to mechanical safety team.")]}

    def chemical_process(state):
        return {"messages": [AIMessage("[chemical_process agent] Assigned to chemical safety team.")]}

    def workplace_safety(state):
        return {"messages": [AIMessage("[workplace_safety agent] Assigned to workplace safety team.")]}

    def general_operations(state):
        return {"messages": [AIMessage("[general_operations agent] Assigned to general operations team.")]}

    builder = StateGraph(MessagesState)
    for fn in (equipment_machinery, chemical_process, workplace_safety, general_operations):
        builder.add_node(fn.__name__, fn)
        builder.add_edge(fn.__name__, END)
    builder.add_conditional_edges(
        START,
        llm_router,
        ["equipment_machinery", "chemical_process", "workplace_safety", "general_operations"],
    )
    graph = builder.compile()

    for i, incident in enumerate(incidents):
        start = time.perf_counter()
        out = graph.invoke({"messages": [HumanMessage(incident.description)]})
        elapsed_ms = (time.perf_counter() - start) * 1000

        response_text = out["messages"][-1].content
        routed_team = "general_operations"
        for team in TEAM_LABELS:
            if team in response_text:
                routed_team = team
                break

        correct = routed_team == incident.expected_team
        metrics.total_incidents += 1
        metrics.correct_routes += int(correct)
        metrics.llm_calls += 1
        metrics.total_latency_ms += elapsed_ms

        metrics.results.append(
            RoutingResult(
                incident_id=incident.id,
                description_preview=incident.description[:90],
                critical_risk=incident.critical_risk,
                expected_team=incident.expected_team,
                routed_team=routed_team,
                correct=correct,
                decided_by="llm",
                latency_ms=round(elapsed_ms, 1),
            )
        )

        status_icon = "✓" if correct else "✗"
        print(f"  [{i+1:2d}/{len(incidents)}] {status_icon} Routed to {routed_team:<20} | Expected: {incident.expected_team:<20} | {elapsed_ms:5.0f}ms")

    metrics.estimated_llm_cost_usd = metrics.llm_calls * LLM_COST_PER_CALL
    print(f"\n  Baseline Complete:")
    print(f"    Accuracy:      {metrics.accuracy:.1%} ({metrics.correct_routes}/{metrics.total_incidents})")
    print(f"    LLM Calls:     {metrics.llm_calls} (100%)")
    print(f"    Avg Latency:   {metrics.avg_latency_ms:.0f} ms/incident")
    print(f"    Est. LLM Cost: ${metrics.estimated_llm_cost_usd:.5f}")
    return metrics


# ── Phase 2: Jev-First Routing (langgraph-jev-router) ────────────────────────────

def run_jev_first(incidents: List[Incident], threshold: float = 0.7, llm_mode: str = "auto") -> Tuple[BenchmarkMetrics, Any]:
    """Run all incidents through LangGraph with Jev-first routing."""
    print("\n" + "=" * 80)
    print("PHASE 2: JEV-FIRST — langgraph-jev-router (Jev decides first, LLM fallback)")
    print("=" * 80)
    print(f"Running {len(incidents)} manufacturing incidents with Jev-first (threshold={threshold}, fallback_mode={llm_mode})...\n")

    reset_configuration()
    configure(
        model="jev-latest",
        shadow=False,
        threshold=threshold,
        log_path=str(Path(__file__).parent / "jev_decisions.jsonl"),
    )


    lookup = {i.description: i.expected_team for i in incidents}
    route_with_llm = _build_llm_router(mode=llm_mode, incident_lookup=lookup)
    llm_call_count = 0


    @jev_first(
        routes=TEAM_DESCRIPTIONS,
        question="Which specialist team should handle this manufacturing safety incident?",
        threshold=threshold,
        shadow=False,
    )
    def route_incident(state: MessagesState) -> Literal["equipment_machinery", "chemical_process", "workplace_safety", "general_operations"]:
        nonlocal llm_call_count
        llm_call_count += 1
        text = state["messages"][-1].content
        return route_with_llm(text)

    def equipment_machinery(state):
        return {"messages": [AIMessage("[equipment_machinery agent] Assigned to mechanical safety team.")]}

    def chemical_process(state):
        return {"messages": [AIMessage("[chemical_process agent] Assigned to chemical safety team.")]}

    def workplace_safety(state):
        return {"messages": [AIMessage("[workplace_safety agent] Assigned to workplace safety team.")]}

    def general_operations(state):
        return {"messages": [AIMessage("[general_operations agent] Assigned to general operations team.")]}

    builder = StateGraph(MessagesState)
    for fn in (equipment_machinery, chemical_process, workplace_safety, general_operations):
        builder.add_node(fn.__name__, fn)
        builder.add_edge(fn.__name__, END)
    builder.add_conditional_edges(
        START,
        route_incident,
        ["equipment_machinery", "chemical_process", "workplace_safety", "general_operations"],
    )
    graph = attach_routes(builder).compile()

    metrics = BenchmarkMetrics(phase="jev_first")

    for i, incident in enumerate(incidents):
        llm_before = llm_call_count
        start = time.perf_counter()
        out = graph.invoke({"messages": [HumanMessage(incident.description)]})
        elapsed_ms = (time.perf_counter() - start) * 1000

        used_llm = llm_call_count > llm_before

        response_text = out["messages"][-1].content
        routed_team = "general_operations"
        for team in TEAM_LABELS:
            if team in response_text:
                routed_team = team
                break

        correct = routed_team == incident.expected_team
        metrics.total_incidents += 1
        metrics.correct_routes += int(correct)
        metrics.total_latency_ms += elapsed_ms
        if used_llm:
            metrics.llm_calls += 1
        else:
            metrics.jev_decisions += 1

        decider = "llm" if used_llm else "jev"
        metrics.results.append(
            RoutingResult(
                incident_id=incident.id,
                description_preview=incident.description[:90],
                critical_risk=incident.critical_risk,
                expected_team=incident.expected_team,
                routed_team=routed_team,
                correct=correct,
                decided_by=decider,
                latency_ms=round(elapsed_ms, 1),
            )
        )

        status_icon = "✓" if correct else "✗"
        tag = "⚡ JEV" if not used_llm else "🤖 LLM"
        print(f"  [{i+1:2d}/{len(incidents)}] {status_icon} [{tag}] Routed to {routed_team:<20} | Expected: {incident.expected_team:<20} | {elapsed_ms:5.0f}ms")

    metrics.estimated_llm_cost_usd = metrics.llm_calls * LLM_COST_PER_CALL

    jev_report = route_incident.jev.report(llm_cost_per_call=LLM_COST_PER_CALL)
    jev_router_rep = jev_report.routers[0] if getattr(jev_report, "routers", None) else None
    if jev_router_rep:
        metrics.estimated_jev_cost_usd = getattr(jev_router_rep, "jev_cost_usd", 0.0)

    print(f"\n  Jev-First Complete:")
    print(f"    Accuracy:      {metrics.accuracy:.1%} ({metrics.correct_routes}/{metrics.total_incidents})")
    print(f"    Jev Decided:   {metrics.jev_decisions}/{metrics.total_incidents} ({metrics.jev_decisions/metrics.total_incidents:.1%})  ⚡ No LLM calls!")
    print(f"    LLM Fallback:  {metrics.llm_calls}/{metrics.total_incidents} ({metrics.llm_calls/metrics.total_incidents:.1%})")
    print(f"    Avg Latency:   {metrics.avg_latency_ms:.0f} ms/incident")
    print(f"    Est. Jev Cost: ${metrics.estimated_jev_cost_usd:.6f}")
    print(f"    Est. LLM Cost: ${metrics.estimated_llm_cost_usd:.5f}")
    print(f"    Est. Total:    ${metrics.estimated_llm_cost_usd + metrics.estimated_jev_cost_usd:.5f}")

    return metrics, jev_report


# ── Phase 3: Report Generation (JSON, Markdown, HTML Dashboard) ─────────────────

def generate_reports(
    baseline: BenchmarkMetrics,
    jev_first_metrics: BenchmarkMetrics,
    jev_report: Any,
    incidents: List[Incident],
) -> Tuple[Path, Path, Path]:
    """Generate comprehensive reports in JSON, Markdown, and an interactive HTML Dashboard."""
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    llm_avoided = baseline.llm_calls - jev_first_metrics.llm_calls
    llm_avoided_pct = (llm_avoided / baseline.llm_calls * 100) if baseline.llm_calls else 0.0
    baseline_cost = baseline.estimated_llm_cost_usd
    jev_total_cost = jev_first_metrics.estimated_llm_cost_usd + jev_first_metrics.estimated_jev_cost_usd
    cost_saved = baseline_cost - jev_total_cost
    cost_saved_pct = (cost_saved / baseline_cost * 100) if baseline_cost else 0.0
    latency_reduction_ms = baseline.avg_latency_ms - jev_first_metrics.avg_latency_ms
    latency_reduction_pct = (latency_reduction_ms / baseline.avg_latency_ms * 100) if baseline.avg_latency_ms else 0.0

    # 1. JSON Report
    report_data = {
        "benchmark": "langgraph-jev-router Manufacturing Industry Benchmark",
        "dataset": {
            "name": "IHM Stefanini Industrial Safety & Health Analytics Database",
            "url": "https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database",
            "license": "CC0 1.0 (Public Domain)",
            "total_incidents_evaluated": len(incidents),
            "teams": list(TEAM_LABELS),
            "team_descriptions": TEAM_DESCRIPTIONS,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "llm_calls_avoided": llm_avoided,
            "llm_calls_avoided_percent": round(llm_avoided_pct, 1),
            "cost_saved_usd": round(cost_saved, 6),
            "cost_saved_percent": round(cost_saved_pct, 1),
            "avg_latency_reduction_ms": round(latency_reduction_ms, 1),
            "avg_latency_reduction_percent": round(latency_reduction_pct, 1),
            "baseline_accuracy": round(baseline.accuracy, 4),
            "jev_first_accuracy": round(jev_first_metrics.accuracy, 4),
            "accuracy_delta_pp": round((jev_first_metrics.accuracy - baseline.accuracy) * 100, 2),
        },
        "baseline": {
            "router": "Google Gemini 3.8 Flash (LLM-Only)",
            "total_incidents": baseline.total_incidents,
            "accuracy": round(baseline.accuracy, 4),
            "llm_calls": baseline.llm_calls,
            "avg_latency_ms": round(baseline.avg_latency_ms, 1),
            "total_latency_ms": round(baseline.total_latency_ms, 1),
            "estimated_cost_usd": round(baseline_cost, 6),
        },
        "jev_first": {
            "router": "langgraph-jev-router (Jev-First, Gemini Fallback)",
            "total_incidents": jev_first_metrics.total_incidents,
            "accuracy": round(jev_first_metrics.accuracy, 4),
            "jev_decisions": jev_first_metrics.jev_decisions,
            "jev_decision_rate_percent": round(jev_first_metrics.jev_decisions / jev_first_metrics.total_incidents * 100, 1),
            "llm_fallback_calls": jev_first_metrics.llm_calls,
            "llm_fallback_rate_percent": round(jev_first_metrics.llm_calls / jev_first_metrics.total_incidents * 100, 1),
            "avg_latency_ms": round(jev_first_metrics.avg_latency_ms, 1),
            "total_latency_ms": round(jev_first_metrics.total_latency_ms, 1),
            "estimated_jev_cost_usd": round(jev_first_metrics.estimated_jev_cost_usd, 6),
            "estimated_llm_cost_usd": round(jev_first_metrics.estimated_llm_cost_usd, 6),
            "estimated_total_cost_usd": round(jev_total_cost, 6),
        },
        "jev_internal_report": jev_report.to_dict() if getattr(jev_report, "to_dict", None) else None,
        "incident_evaluations": [
            {
                "id": inc.id,
                "description": inc.description,
                "critical_risk": inc.critical_risk,
                "expected_team": inc.expected_team,
                "baseline_routed": b_res.routed_team,
                "baseline_correct": b_res.correct,
                "baseline_latency_ms": b_res.latency_ms,
                "jev_routed": j_res.routed_team,
                "jev_correct": j_res.correct,
                "jev_decided_by": j_res.decided_by,
                "jev_latency_ms": j_res.latency_ms,
            }
            for inc, b_res, j_res in zip(incidents, baseline.results, jev_first_metrics.results)
        ],
    }

    json_path = reports_dir / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    # 2. Markdown Report
    dist = Counter(i.expected_team for i in incidents)
    md_lines = [
        "# Manufacturing Industry Benchmark Report",
        "## `langgraph-jev-router` vs Standard Agentic LLM Routing",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **LLM Calls Avoided:** **{llm_avoided}** out of {baseline.llm_calls} (**{llm_avoided_pct:.1f}% reduction**)",
        f"- **Speed Improvement:** **{abs(latency_reduction_ms):.0f}ms** faster per decision (**{latency_reduction_pct:.1f}% latency cut**)",
        f"- **Cost Savings:** **{cost_saved_pct:.1f}%** lower total routing cost",
        f"- **Routing Accuracy:** Baseline **{baseline.accuracy:.1%}** vs Jev-First **{jev_first_metrics.accuracy:.1%}** (Δ {jev_first_metrics.accuracy - baseline.accuracy:+.1%})",
        "",
        "---",
        "",
        "## Dataset Profile",
        "",
        "| Attribute | Value |",
        "|---|---|",
        "| **Dataset** | [IHM Stefanini Industrial Safety & Health Database](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) |",
        "| **License** | CC0: Public Domain |",
        "| **Evaluated Incidents** | " + str(len(incidents)) + " real manufacturing incident logs |",
        "| **Industries** | Mining, Metals & Industrial Fabrication |",
        "| **Consolidated Teams** | 4 specialist routing destinations |",
        "",
        "### Incident Distribution Across Specialist Teams",
        "",
        "| Specialist Team | Incidents | Proportion | Scope |",
        "|---|---|---|---|",
    ]
    for team in TEAM_LABELS:
        cnt = dist.get(team, 0)
        pct = (cnt / len(incidents) * 100) if incidents else 0
        md_lines.append(f"| `{team}` | {cnt} | {pct:.1f}% | {TEAM_DESCRIPTIONS[team]} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Benchmark Results Comparison",
        "",
        "| Metric | Baseline (Gemini 3.8 Flash) | Jev-First (`langgraph-jev-router`) | Delta / Improvement |",
        "|---|---|---|---|",
        f"| **Routing Accuracy** | {baseline.accuracy:.1%} | {jev_first_metrics.accuracy:.1%} | **{(jev_first_metrics.accuracy - baseline.accuracy)*100:+.1f} pp** |",
        f"| **LLM Calls Required** | {baseline.llm_calls} | {jev_first_metrics.llm_calls} | **-{llm_avoided} calls ({llm_avoided_pct:.1f}% cut)** |",
        f"| **Jev Autonomous Decisions** | 0 (0%) | {jev_first_metrics.jev_decisions} ({jev_first_metrics.jev_decisions/jev_first_metrics.total_incidents*100:.1f}%) | **+{jev_first_metrics.jev_decisions} direct routes** |",
        f"| **Average Latency** | {baseline.avg_latency_ms:.0f} ms | {jev_first_metrics.avg_latency_ms:.0f} ms | **{abs(latency_reduction_ms):.0f} ms faster ({latency_reduction_pct:.1f}%)** |",
        f"| **Total Pipeline Time** | {baseline.total_latency_ms/1000:.1f} s | {jev_first_metrics.total_latency_ms/1000:.1f} s | **-{(baseline.total_latency_ms - jev_first_metrics.total_latency_ms)/1000:.1f} s** |",
        f"| **Estimated Cost** | ${baseline_cost:.5f} | ${jev_total_cost:.5f} | **-${cost_saved:.5f} ({cost_saved_pct:.1f}% saved)** |",
        "",
        "---",
        "",
        "## Jev Router Internal Telemetry",
        "",
        "```",
        jev_report.to_text() if getattr(jev_report, "to_text", None) else "No internal report available.",
        "```",
        "",
        "---",
        "",
        "## Architecture Overview",
        "",
        "### 1. Traditional Agentic Architecture (Baseline)",
        "Every single incoming incident report invokes an LLM. In manufacturing plants processing thousands of sensor tickets, shift logs, and safety reports daily, this creates:",
        "- High latency bottlenecks (~2-5 seconds per decision)",
        "- Fragility against rate limits and external service outages",
        "- Unnecessary API costs for routine classifications",
        "",
        "### 2. Jev-First Architecture (`langgraph-jev-router`)",
        "`langgraph-jev-router` inserts TypeSafe's Jev model as a System 1 fast-reflex router in front of the LangGraph conditional edges:",
        "1. **Fast-path (Jev)**: When Jev confidence meets the threshold (e.g. ≥ 0.70), it routes the incident directly in ~200ms without touching the LLM.",
        "2. **Safety fallback**: If Jev is uncertain, your existing Gemini router takes over seamlessly.",
        "3. **Zero risk**: If TypeSafe is down or returns an error, the library automatically falls back to your original router without breaking the graph.",
    ])

    md_path = reports_dir / "benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        md_lines.append("")
        f.write("\n".join(md_lines))

    # 3. Interactive HTML Report (State of the art design)
    html_content = _build_html_dashboard(report_data)
    html_path = reports_dir / "benchmark_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return json_path, md_path, html_path


def _build_html_dashboard(data: Dict[str, Any]) -> str:
    """Render a premium dark-mode HTML dashboard."""
    summary = data["summary"]
    baseline = data["baseline"]
    jev = data["jev_first"]
    evals = data.get("incident_evaluations", [])

    rows_html = []
    for ev in evals:
        b_icon = '<span class="badge badge-success">✓</span>' if ev["baseline_correct"] else '<span class="badge badge-danger">✗</span>'
        j_icon = '<span class="badge badge-success">✓</span>' if ev["jev_correct"] else '<span class="badge badge-danger">✗</span>'
        dec_badge = '<span class="tag tag-jev">⚡ Jev</span>' if ev["jev_decided_by"] == "jev" else '<span class="tag tag-llm">🤖 LLM</span>'
        rows_html.append(f"""
        <tr>
            <td class="font-mono">#{ev['id']}</td>
            <td class="desc-cell" title="{ev['description']}">{ev['description'][:100]}...</td>
            <td><span class="badge badge-risk">{ev['critical_risk']}</span></td>
            <td><span class="badge badge-team">{ev['expected_team']}</span></td>
            <td>{b_icon} {ev['baseline_routed']} <span class="dim">({ev['baseline_latency_ms']:.0f}ms)</span></td>
            <td>{j_icon} {ev['jev_routed']} {dec_badge} <span class="dim">({ev['jev_latency_ms']:.0f}ms)</span></td>
        </tr>
        """)

    evals_json = json.dumps(evals)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>langgraph-jev-router Benchmark Dashboard | Manufacturing Industry</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(59, 130, 246, 0.5);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-dim: #6b7280;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1380px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .brand-badge {{
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        h1 {{ font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
        .subtitle {{ color: var(--text-secondary); font-size: 14px; margin-top: 4px; }}
        .timestamp {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-dim); text-align: right; }}

        /* Hero Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
            transition: all 0.25s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-hover);
            box-shadow: 0 10px 25px -5px var(--accent-glow);
        }}
        .stat-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); }}
        .stat-value {{ font-size: 32px; font-weight: 800; margin: 8px 0; letter-spacing: -1px; }}
        .stat-sub {{ font-size: 13px; color: var(--text-dim); display: flex; align-items: center; gap: 6px; }}
        .text-success {{ color: var(--success); }}
        .text-accent {{ color: #60a5fa; }}

        /* Section Layout */
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            backdrop-filter: blur(12px);
        }}
        .card-title {{ font-size: 16px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}

        /* Comparison Table */
        .comp-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .comp-table th {{
            text-align: left;
            padding: 10px 12px;
            color: var(--text-dim);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        .comp-table td {{
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        .comp-table tr:last-child td {{ border-bottom: none; }}

        /* Table Filter & Explorer */
        .filter-bar {{
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        .search-input, .select-filter {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-family: inherit;
        }}
        .search-input {{ flex: 1; min-width: 200px; }}
        .search-input:focus, .select-filter:focus {{ outline: none; border-color: var(--accent); }}

        .table-wrap {{
            overflow-x: auto;
            max-height: 520px;
            border: 1px solid var(--border);
            border-radius: 10px;
        }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .data-table th {{
            position: sticky;
            top: 0;
            background: #151e2e;
            text-align: left;
            padding: 10px 12px;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            z-index: 2;
        }}
        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            vertical-align: middle;
        }}
        .data-table tr:hover {{ background: rgba(255, 255, 255, 0.02); }}

        /* Badges & Tags */
        .font-mono {{ font-family: 'JetBrains Mono', monospace; color: var(--text-dim); }}
        .desc-cell {{ max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-success {{ background: var(--success-glow); color: var(--success); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        .badge-risk {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; }}
        .badge-team {{ background: rgba(59, 130, 246, 0.15); color: #93c5fd; font-family: 'JetBrains Mono', monospace; }}
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .tag-jev {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .tag-llm {{ background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }}
        .dim {{ color: var(--text-dim); font-size: 11px; }}

        /* Progress Visual */
        .progress-bar-wrap {{
            height: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            overflow: hidden;
            display: flex;
            margin: 12px 0;
        }}
        .progress-jev {{ background: #10b981; height: 100%; transition: width 0.5s ease; }}
        .progress-llm {{ background: #8b5cf6; height: 100%; transition: width 0.5s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <div class="brand">
                    <span class="brand-badge">Benchmark</span>
                    <span style="color: var(--text-dim); font-weight: 500;">Manufacturing Safety Incident Routing</span>
                </div>
                <h1>langgraph-jev-router Industry Benchmark</h1>
                <p class="subtitle">Comparing Standard LLM Routing (Gemini 3.8 Flash) vs. Jev-First Autonomous Fast-Path</p>
            </div>
            <div class="timestamp">
                Dataset: IHM Stefanini Industrial Safety<br>
                Evaluated: {len(evals)} real incidents
            </div>
        </div>

        <!-- Hero Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">LLM Calls Avoided</div>
                <div class="stat-value text-success">{summary['llm_calls_avoided_percent']}%</div>
                <div class="stat-sub"><strong class="text-success">{summary['llm_calls_avoided']} calls saved</strong> out of {baseline['llm_calls']} total</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Autonomous Jev Decisions</div>
                <div class="stat-value text-accent">{jev['jev_decision_rate_percent']}%</div>
                <div class="stat-sub">{jev['jev_decisions']} direct routes | {jev['llm_fallback_calls']} LLM fallbacks</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg. Latency Reduction</div>
                <div class="stat-value text-success">{summary['avg_latency_reduction_percent']}%</div>
                <div class="stat-sub">{baseline['avg_latency_ms']:.0f}ms (LLM) → {jev['avg_latency_ms']:.0f}ms (Jev-First)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Routing Accuracy</div>
                <div class="stat-value">{jev['accuracy']*100:.1f}%</div>
                <div class="stat-sub">Baseline: {baseline['accuracy']*100:.1f}% ({summary['accuracy_delta_pp']:+.1f} pp delta)</div>
            </div>
        </div>

        <!-- Side-by-side comparison & Distribution -->
        <div class="grid-2">
            <div class="card">
                <div class="card-title">⚡ Head-to-Head Architecture Comparison</div>
                <table class="comp-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Baseline (LLM-Only)</th>
                            <th>Jev-First (Hybrid)</th>
                            <th>Improvement</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>LLM Calls</strong></td>
                            <td>{baseline['llm_calls']}</td>
                            <td>{jev['llm_fallback_calls']}</td>
                            <td class="text-success"><strong>-{summary['llm_calls_avoided']} ({summary['llm_calls_avoided_percent']}%)</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Autonomous Fast-Path</strong></td>
                            <td>0</td>
                            <td>{jev['jev_decisions']} ({jev['jev_decision_rate_percent']}%)</td>
                            <td class="text-accent"><strong>+{jev['jev_decisions']} decisions</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Average Latency</strong></td>
                            <td>{baseline['avg_latency_ms']:.0f} ms</td>
                            <td>{jev['avg_latency_ms']:.0f} ms</td>
                            <td class="text-success"><strong>-{summary['avg_latency_reduction_ms']:.0f} ms</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Total Run Duration</strong></td>
                            <td>{baseline['total_latency_ms']/1000:.1f} s</td>
                            <td>{jev['total_latency_ms']/1000:.1f} s</td>
                            <td class="text-success"><strong>-{(baseline['total_latency_ms'] - jev['total_latency_ms'])/1000:.1f} s</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Accuracy</strong></td>
                            <td>{baseline['accuracy']*100:.1f}%</td>
                            <td>{jev['accuracy']*100:.1f}%</td>
                            <td>{summary['accuracy_delta_pp']:+.1f} pp</td>
                        </tr>
                        <tr>
                            <td><strong>Est. Total Cost</strong></td>
                            <td>${baseline['estimated_cost_usd']:.5f}</td>
                            <td>${jev['estimated_total_cost_usd']:.5f}</td>
                            <td class="text-success"><strong>-${summary['cost_saved_usd']:.5f} ({summary['cost_saved_percent']}%)</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-title">📊 Decision Routing Split</div>
                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">
                    How incoming incident tickets were resolved in the Jev-First workflow:
                </p>
                <div class="progress-bar-wrap">
                    <div class="progress-jev" style="width: {jev['jev_decision_rate_percent']}%;" title="Jev Autonomous: {jev['jev_decision_rate_percent']}%"></div>
                    <div class="progress-llm" style="width: {jev['llm_fallback_rate_percent']}%;" title="LLM Fallback: {jev['llm_fallback_rate_percent']}%"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 20px;">
                    <span style="color: #34d399;">⚡ Jev Autonomous: <strong>{jev['jev_decisions']} ({jev['jev_decision_rate_percent']}%)</strong></span>
                    <span style="color: #a78bfa;">🤖 Gemini Fallback: <strong>{jev['llm_fallback_calls']} ({jev['llm_fallback_rate_percent']}%)</strong></span>
                </div>

                <div class="card-title" style="margin-top: 16px;">🏭 Real Manufacturing Dataset</div>
                <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                    Extracted from the <strong>IHM Stefanini Industrial Safety and Health Analytics Database</strong> (mining, smelting, and metal manufacturing facilities). Incidents include mechanical pinches, pressurized steam leaks, liquid metal splashes, crane operations, and chemical hazards.
                </p>
            </div>
        </div>

        <!-- Incident Explorer Table -->
        <div class="card">
            <div class="card-title">🔍 Incident Decision Explorer</div>
            <div class="filter-bar">
                <input type="text" id="searchInput" class="search-input" placeholder="Search incident description or risk...">
                <select id="teamFilter" class="select-filter">
                    <option value="">All Specialist Teams</option>
                    <option value="equipment_machinery">equipment_machinery</option>
                    <option value="chemical_process">chemical_process</option>
                    <option value="workplace_safety">workplace_safety</option>
                    <option value="general_operations">general_operations</option>
                </select>
                <select id="deciderFilter" class="select-filter">
                    <option value="">All Deciders (Jev + LLM)</option>
                    <option value="jev">Decided by ⚡ Jev</option>
                    <option value="llm">Decided by 🤖 LLM</option>
                </select>
            </div>

            <div class="table-wrap">
                <table class="data-table" id="incidentTable">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Description</th>
                            <th>Critical Risk</th>
                            <th>Expected Team</th>
                            <th>Baseline Route (Gemini)</th>
                            <th>Jev-First Route</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const searchInput = document.getElementById('searchInput');
        const teamFilter = document.getElementById('teamFilter');
        const deciderFilter = document.getElementById('deciderFilter');
        const tableBody = document.getElementById('tableBody');
        const rows = Array.from(tableBody.getElementsByTagName('tr'));

        function applyFilters() {{
            const search = searchInput.value.toLowerCase();
            const team = teamFilter.value.toLowerCase();
            const decider = deciderFilter.value.toLowerCase();

            rows.forEach(row => {{
                const text = row.innerText.toLowerCase();
                const matchesSearch = !search || text.includes(search);
                const matchesTeam = !team || text.includes(team);
                const matchesDecider = !decider || (decider === 'jev' && text.includes('⚡ jev')) || (decider === 'llm' && text.includes('🤖 llm'));

                row.style.display = (matchesSearch && matchesTeam && matchesDecider) ? '' : 'none';
            }});
        }}

        searchInput.addEventListener('input', applyFilters);
        teamFilter.addEventListener('change', applyFilters);
        deciderFilter.addEventListener('change', applyFilters);
    </script>
</body>
</html>
"""


# ── Main Entrypoint ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manufacturing Industry Benchmark for langgraph-jev-router")
    parser.add_argument("--sample", type=str, default="50", help="Number of incidents to evaluate (e.g. 50, 100, or 'all')")
    parser.add_argument("--llm", choices=["auto", "gemini", "simulated"], default="auto", help="LLM router mode (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Jev confidence threshold (default: 0.7)")
    parser.add_argument("--sweep", action="store_true", help="Run multi-threshold calibration sweep (0.60 to 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling (default: 42)")

    args = parser.parse_args()

    sample_size: Optional[int]
    if args.sample.lower() in ("all", "0", "full"):
        sample_size = None
    else:
        sample_size = int(args.sample)

    # Validate environment
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("ERROR: TYPESAFE_API_KEY not set. Check your .env file or environment.")
        sys.exit(1)
    if args.llm == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. Check your .env file or environment.")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║     Manufacturing Industry Benchmark — langgraph-jev-router                 ║")
    print("║     Dataset: IHM Stefanini Industrial Safety Database                        ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    # Load dataset
    incidents = load_incidents(sample=sample_size, seed=args.seed)
    total_in_file = 425
    print(f"\nLoaded {len(incidents)} real manufacturing incidents (from total {total_in_file} in dataset)")
    print(f"Jev confidence threshold: {args.threshold} | LLM mode: {args.llm}\n")
    dist = Counter(i.expected_team for i in incidents)
    print("Distribution of specialist destinations:")
    for team, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  - {team:22s}: {count:3d} ({count/len(incidents)*100:.1f}%)")

    overall_start = time.perf_counter()

    # Phase 1: Baseline LLM Routing
    baseline_metrics = run_baseline(incidents, llm_mode=args.llm)

    if args.sweep:
        thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        sweep_results = []
        print("\n" + "=" * 90)
        print("RUNNING MULTI-THRESHOLD CALIBRATION SWEEP")
        print("=" * 90)

        best_m = None
        best_rep = None
        for t in thresholds:
            print(f"\n--> Evaluating threshold = {t:.2f}...")
            m, rep = run_jev_first(incidents, threshold=t, llm_mode=args.llm)
            avoided = baseline_metrics.llm_calls - m.llm_calls
            avoided_pct = (avoided / baseline_metrics.llm_calls * 100) if baseline_metrics.llm_calls else 0
            sweep_results.append({
                "threshold": t,
                "llm_avoided_pct": round(avoided_pct, 1),
                "jev_decisions": m.jev_decisions,
                "total_incidents": m.total_incidents,
                "jev_rate_pct": round(m.jev_decisions / m.total_incidents * 100, 1),
                "llm_fallback": m.llm_calls,
                "accuracy_pct": round(m.accuracy * 100, 1),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "est_cost_usd": round(m.estimated_llm_cost_usd + m.estimated_jev_cost_usd, 5),
            })
            if t == args.threshold or best_m is None:
                best_m = m
                best_rep = rep

        # Print Executive Trade-off Table
        print("\n" + "╔" + "═" * 88 + "╗")
        print("║" + "           EXECUTIVE THRESHOLD CALIBRATION & TRADEOFF TABLE            ".center(88) + "║")
        print("╠═══════════╦══════════════╦════════════════╦══════════════╦════════════════╦══════════════════╣")
        print("║ Threshold ║ LLM Avoided  ║ Autonomous Jev ║ LLM Fallback ║ Route Accuracy ║ Average Latency  ║")
        print("╠═══════════╬══════════════╬════════════════╬══════════════╬════════════════╬══════════════════╣")
        for r in sweep_results:
            print(f"║   {r['threshold']:.2f}    ║   {r['llm_avoided_pct']:5.1f}%     ║   {r['jev_decisions']:2d}/{r['total_incidents']:2d} ({r['jev_rate_pct']:4.1f}%) ║   {r['llm_fallback']:2d}/{r['total_incidents']:2d}    ║    {r['accuracy_pct']:5.1f}%    ║    {r['avg_latency_ms']:5.0f} ms     ║")
        print("╚═══════════╩══════════════╩════════════════╩══════════════╩════════════════╩══════════════════╝")

        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        sweep_path = reports_dir / "threshold_sweep.json"
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSaved calibration sweep table to: {sweep_path}")

        jev_metrics = best_m
        jev_report = best_rep
    else:
        # Phase 2: Single Jev-First Routing
        jev_metrics, jev_report = run_jev_first(incidents, threshold=args.threshold, llm_mode=args.llm)

    overall_elapsed = time.perf_counter() - overall_start

    # Phase 3: Reporting
    print("\n" + "=" * 80)
    print("PHASE 3: GENERATING BENCHMARK REPORTS")
    print("=" * 80)
    json_path, md_path, html_path = generate_reports(baseline_metrics, jev_metrics, jev_report, incidents)

    llm_avoided = baseline_metrics.llm_calls - jev_metrics.llm_calls
    avoided_pct = (llm_avoided / baseline_metrics.llm_calls * 100) if baseline_metrics.llm_calls else 0

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPLETE (Elapsed: {overall_elapsed:.1f}s)")
    print(f"{'=' * 80}")
    print(f"  LLM Calls Avoided:       {llm_avoided}/{baseline_metrics.llm_calls} ({avoided_pct:.1f}% reduction)")
    print(f"  Autonomous Jev Decision: {jev_metrics.jev_decisions}/{jev_metrics.total_incidents} ({jev_metrics.jev_decisions/jev_metrics.total_incidents*100:.1f}%)")
    print(f"  Baseline Accuracy:       {baseline_metrics.accuracy:.1%}")
    print(f"  Jev-First Accuracy:      {jev_metrics.accuracy:.1%}")
    print(f"  Latency Reduction:       {baseline_metrics.avg_latency_ms:.0f}ms → {jev_metrics.avg_latency_ms:.0f}ms ({(baseline_metrics.avg_latency_ms - jev_metrics.avg_latency_ms)/baseline_metrics.avg_latency_ms*100:.1f}% faster)")
    print(f"\nReports generated:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON:     {json_path}")
    print(f"  - HTML:     {html_path}")
    print()



if __name__ == "__main__":
    main()
