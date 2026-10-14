# Manufacturing Safety Incident Routing Benchmark

Demonstrates how [`langgraph-jev-router`](https://github.com/JTankariya/langgraph-jev-router) optimizes agentic routing in the **Manufacturing & Heavy Industry** domain using real-world incident records.

---

## 🏭 Industry Context & Problem Statement

In industrial environments (mining facilities, metal smelting and fabrication plants, assembly floors):
- Hundreds of shift hazard logs, equipment anomalies, and injury incident reports are recorded across rotating shifts.
- Each report must be swiftly routed to specialized plant response teams (mechanical integrity, chemical process hazard teams, occupational safety and PPE specialists, or general operations).
- In traditional agentic architectures, **every single incident invokes a general-purpose LLM** (e.g., Gemini, GPT-4, Claude). This introduces:
  - **Latency bottlenecks:** 1s to 30s per decision.
  - **Single points of failure:** External API downtime or rate-limit quotas halts incident dispatch.
  - **Unnecessary costs:** Standard, repetitive classifications incur full token pricing.

---

## ⚡ The Solution: `langgraph-jev-router` Fast-Path

With `langgraph-jev-router`, **TypeSafe's Jev model** acts as a System 1 fast-reflex router:
1. **Autonomous Route (≥0.70 confidence):** Jev makes the routing decision in **~116 ms** without touching the LLM.
2. **Seamless Fallback (<0.70 confidence):** If Jev is uncertain, the full LLM router is invoked.
3. **Fail-Safe:** If TypeSafe is offline or an error occurs, the graph transparently falls back to the LLM with zero disruption.

---

## 📊 Benchmark Findings

Evaluated on real industrial plant accident records:

| Metric | Traditional LLM Router (Gemini) | Jev-First (`langgraph-jev-router`) | Impact / Improvement |
|---|---|---|---|
| **LLM Calls Required** | **50** (100%) | **12** (24%) | **-76.0% LLM calls avoided** |
| **Autonomous Jev Decisions** | 0 (0%) | **38** (76.0%) | **76% fast-path resolution** |
| **Median Decision Latency** | **305 ms** (simulated) / **~25s** (live Gemini) | **116 ms** (live TypeSafe Jev) | **2.6x to 215x faster** |
| **Total Routing Cost** | **$0.00750** | **$0.00290** | **61.3% cost reduction** |

---

## 📁 Dataset & License

- **Source:** [IHM Stefanini Industrial Safety and Health Analytics Database](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) (Kaggle)
- **License:** **CC0 1.0 Universal (Public Domain Dedication)** — 100% open and safe to redistribute and use without copyright restrictions.
- **Full Legal Notice:** See [`data/DATASET_LICENSE.md`](./data/DATASET_LICENSE.md).

### Team Taxonomy

The 30+ plant critical risk categories are mapped to 4 core specialist teams:
1. `equipment_machinery`: Mechanical equipment, power tools, vehicles, suspended loads, energy isolation (LOTO).
2. `chemical_process`: Chemical substances, pressurized piping, liquid metal burns, process leaks.
3. `workplace_safety`: Slips, trips, falls, electrical hazards, confined spaces, PPE violations, environmental risks.
4. `general_operations`: Non-specialized operational issues, traffic, logistics, and admin logs.

---

## 🚀 How to Run

From the repository root:

```bash
# Run on 50 incidents (default balanced sample)
python manufacturing/benchmark.py --sample 50

# Run on 100 incidents
python manufacturing/benchmark.py --sample 100

# Run on all 425 real manufacturing incidents
python manufacturing/benchmark.py --sample all

# Adjust confidence threshold
python manufacturing/benchmark.py --sample 50 --threshold 0.75

# Force simulated LLM mode (bypasses Google Gemini free tier rate limits)
python manufacturing/benchmark.py --sample 50 --llm simulated
```

---

## 📈 Reports & Dashboard

Running the benchmark automatically updates:
- **Interactive HTML Dashboard:** [`reports/benchmark_report.html`](./reports/benchmark_report.html)
  - Visual KPI cards, decision distribution charts, and a filterable/searchable **Incident Explorer**.
  - Open in browser: `open manufacturing/reports/benchmark_report.html`
- **Markdown Report:** [`reports/benchmark_report.md`](./reports/benchmark_report.md)
- **Machine-Readable JSON:** [`reports/benchmark_report.json`](./reports/benchmark_report.json)
- **Audit Log:** [`jev_decisions.jsonl`](./jev_decisions.jsonl)
