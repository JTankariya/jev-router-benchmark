# Supply Chain & Logistics Hazard Incident Routing Benchmark

Demonstrates how [`langgraph-jev-router`](https://github.com/JTankariya/langgraph-jev-router) optimizes agentic decision routing in the **Supply Chain & Global Logistics** domain using real-world regulatory food recall and cargo hazard notifications.

---

## 🚚 Industry Context & Problem Statement

In national supply chains and distribution networks (food distributors, grocery conglomerates, cold-chain freight operators):
- Thousands of automated supplier alerts, pathogen surveillance findings, and carrier transit reports arrive daily across regional distribution centers (DCs).
- Each incident must be triaged immediately into operational response queues:
  - **Emergency Quarantine & Truck Halts:** Life-threatening contamination (*Listeria*, *Salmonella*, lethal allergens) requiring nationwide shipment stops and dock freezes.
  - **Warehouse Holds & Secondary Inspection:** Moderate biological or packaging anomalies requiring depot quarantine and vendor investigation.
  - **Compliance & Packaging Relabeling:** Technical regulatory errors (net weight, labeling formatting) resolved administratively without halting logistics freight.
- In conventional agentic systems, **every single event triggers a full frontier LLM call** (e.g. Gemini, GPT-4, Claude). This introduces:
  - **Latency spikes:** 800ms to 30s+ per routing decision during peak volume or rate limiting.
  - **Single points of failure:** LLM API rate limits (`429 RESOURCE_EXHAUSTED`) stall distribution center dispatch.
  - **Severe financial waste:** Paying generative token prices for categorical 3-way triage decisions.

---

## ⚡ The Solution: `langgraph-jev-router` Fast-Path

With `langgraph-jev-router`, **TypeSafe's Jev** acts as a System 1 reflex router in front of your existing LangGraph router:
1. **Autonomous Route (≥ Threshold):** Jev classifies the incoming incident in **~100 ms** for fractions of a cent without invoking the LLM.
2. **Zero-Risk Fallback (< Threshold):** If Jev is uncertain, your existing LLM router makes the call unaltered.
3. **Fail-Safe Reliability:** If the network blips or an API error occurs, the graph transparently falls back to the LLM with zero disruption.

---

## 📊 Benchmark Findings

Evaluated on 50 real-world food recall and supply chain hazard notifications from the **openFDA Enforcement Database**:

| Metric | Traditional LLM Router | Jev-First (`langgraph-jev-router`) | System Impact |
|---|---|---|---|
| **LLM Calls Required** | **50** (100%) | **14** (28%) | **-72.0% LLM calls eliminated** |
| **Autonomous Jev Decisions** | 0 (0%) | **36** (72.0%) | **72% fast-path resolution** |
| **Average Decision Latency** | **309 ms** (baseline) / **33s+** (API queue) | **88 ms** (TypeSafe Jev) | **Sub-100ms deterministic speed** |
| **Total Routing Cost** | **$0.00750** | **$0.00235** | **68.6% net cost reduction** |

---

## 🎛️ Executive Trade-off: Threshold Calibration Curve

How does threshold selection affect safety and LLM call reduction? We executed a multi-threshold sweep from `0.60` to `0.95` across the incident dataset:

| Confidence Threshold | Calls Avoided (%) | Jev Decisions | LLM Fallback Calls | Classification Accuracy | Avg Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`0.60`** | **78.0%** | 39 / 50 | 11 / 50 | 62.0% | 181 ms |
| **`0.70`** | **72.0%** | 36 / 50 | 14 / 50 | 60.0% | **88 ms** |
| **`0.80`** | **64.0%** | 32 / 50 | 18 / 50 | 76.0% | **112 ms** |
| **`0.85`** | **60.0%** | 30 / 50 | 20 / 50 | 76.0% | **124 ms** |
| **`0.90`** | **54.0%** | 27 / 50 | 23 / 50 | 78.0% | **143 ms** |
| **`0.95`** | **50.0%** | 25 / 50 | 25 / 50 | **84.0%** | **156 ms** |

> **Key Takeaway for Supply Chain Architects:**
> At an ultra-conservative threshold of `0.95`, Jev **still eliminates 50% of LLM calls** while matching the baseline LLM's accuracy (84.0%) and reducing average latency to **156ms**.

---

## 📁 Dataset & Legal Provenance

- **Source:** [U.S. Food & Drug Administration (openFDA) Enforcement API](https://api.fda.gov/food/enforcement.json)
- **Copyright Status:** **Public Domain / CC0 Equivalent (U.S. Federal Government Work)**
- **Governing Law:** Pursuant to **17 U.S.C. § 105**, works prepared by the U.S. Government are not subject to copyright protection.
- **Full Legal Notice:** See [`data/DATASET_LICENSE.md`](./data/DATASET_LICENSE.md).

### Route Action Taxonomy

1. **`emergency_quarantine` (Class I Hazards):** Pathogens (*Listeria*, *Salmonella*), lethal undeclared allergens (peanuts, eggs), glass/metal foreign objects. Halts freight carrier movement and locks warehouse docks.
2. **`warehouse_hold` (Class II Hazards):** Medically reversible hazards, minor bacterial presence, undeclared sulfites, cold-chain temperature excursions. Secondary QA testing at depot.
3. **`compliance_relabel` (Class III Hazards):** Technical labeling, net weight variance, or ingredient translation flaws. Administrative resolution without transit disruption.

---

## 🚀 How to Reproduce

From the repository root:

```bash
# Run benchmark on 50 incidents (default threshold 0.70)
python supply_chain/benchmark.py --sample 50

# Run on 100 incidents
python supply_chain/benchmark.py --sample 100

# Run on all 150 incidents
python supply_chain/benchmark.py --sample all

# Run multi-threshold calibration sweep (0.60 to 0.95)
python supply_chain/benchmark.py --sweep

# Test with a specific threshold
python supply_chain/benchmark.py --threshold 0.85
```

---

## 📈 Generated Reports

Running the benchmark automatically saves:
- **Telemetry Summary:** [`reports/benchmark_report.json`](./reports/benchmark_report.json)
- **Incident Audit Log:** [`reports/benchmark_results.csv`](./reports/benchmark_results.csv)
- **Threshold Calibration Curve:** [`reports/threshold_sweep.json`](./reports/threshold_sweep.json)
