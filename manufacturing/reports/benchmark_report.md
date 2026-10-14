# Manufacturing Industry Benchmark Report
## `langgraph-jev-router` vs Standard Agentic LLM Routing

**Generated:** 2026-10-14 03:11 UTC

---

## Executive Summary

- **LLM Calls Avoided:** **38** out of 50 (**76.0% reduction**)
- **Speed Improvement:** **796ms** faster per decision (**80.8% latency cut**)
- **Cost Savings:** **61.3%** lower total routing cost
- **Routing Accuracy:** Baseline **98.0%** vs Jev-First **58.0%** (Δ -40.0%)

---

## Dataset Profile

| Attribute | Value |
|---|---|
| **Dataset** | [IHM Stefanini Industrial Safety & Health Database](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) |
| **License** | CC0: Public Domain |
| **Evaluated Incidents** | 50 real manufacturing incident logs |
| **Industries** | Mining, Metals & Industrial Fabrication |
| **Consolidated Teams** | 4 specialist routing destinations |

### Incident Distribution Across Specialist Teams

| Specialist Team | Incidents | Proportion | Scope |
|---|---|---|---|
| `equipment_machinery` | 9 | 18.0% | Incidents involving mechanical equipment, tools, vehicles, suspended loads, projections, and machine/energy isolation failures. |
| `chemical_process` | 3 | 6.0% | Incidents involving chemical substances, pressurized systems, liquid metals, burns from process materials, and process spills or leaks. |
| `workplace_safety` | 8 | 16.0% | Incidents involving falls, slips, electrical hazards, confined spaces, PPE failures, and environmental/biological hazards (animals, bees). |
| `general_operations` | 30 | 60.0% | General operational incidents, traffic, administrative issues, and any incidents that don't fit the other specialized categories. |

---

## Benchmark Results Comparison

| Metric | Baseline (Gemini 3.8 Flash) | Jev-First (`langgraph-jev-router`) | Delta / Improvement |
|---|---|---|---|
| **Routing Accuracy** | 98.0% | 58.0% | **-40.0 pp** |
| **LLM Calls Required** | 50 | 12 | **-38 calls (76.0% cut)** |
| **Jev Autonomous Decisions** | 0 (0%) | 38 (76.0%) | **+38 direct routes** |
| **Average Latency** | 985 ms | 189 ms | **796 ms faster (80.8%)** |
| **Total Pipeline Time** | 49.3 s | 9.5 s | **-39.8 s** |
| **Estimated Cost** | $0.00750 | $0.00290 | **-$0.00460 (61.3% saved)** |

---

## Jev Router Internal Telemetry

```
Jev router report: run_jev_first.<locals>.route_incident
--------------------------------------------------------
LIVE   50 routing decisions
  Decided by Jev ........    38 (76%)   <- LLM routing calls avoided
  Sent to your router ...    12 (24%)
      low confidence: 12
  Median latency: Jev 102 ms  vs  your router 305 ms
  Estimated routing time saved: 7.8 s
  Jev cost (est.): $0.001099   LLM routing cost avoided (est.): $0.0057
```

---

## Architecture Overview

### 1. Traditional Agentic Architecture (Baseline)
Every single incoming incident report invokes an LLM. In manufacturing plants processing thousands of sensor tickets, shift logs, and safety reports daily, this creates:
- High latency bottlenecks (~2-5 seconds per decision)
- Fragility against rate limits and external service outages
- Unnecessary API costs for routine classifications

### 2. Jev-First Architecture (`langgraph-jev-router`)
`langgraph-jev-router` inserts TypeSafe's Jev model as a System 1 fast-reflex router in front of the LangGraph conditional edges:
1. **Fast-path (Jev)**: When Jev confidence meets the threshold (e.g. ≥ 0.70), it routes the incident directly in ~200ms without touching the LLM.
2. **Safety fallback**: If Jev is uncertain, your existing Gemini router takes over seamlessly.
3. **Zero risk**: If TypeSafe is down or returns an error, the library automatically falls back to your original router without breaking the graph.
