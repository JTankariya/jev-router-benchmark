# `jev-router-benchmark` ⚡

Industry-specific benchmarks demonstrating how [**`langgraph-jev-router`**](https://github.com/JTankariya/langgraph-jev-router) optimizes agentic workflows by cutting LLM routing calls, reducing latency, and saving cost using real-world enterprise datasets.

---

## 🎯 Overview

Most agentic AI architectures rely on an LLM at every conditional edge to route state to specialized worker agents. While effective, this creates substantial drawbacks in enterprise deployments:
- **High latency bottlenecks** (300ms to 30+ seconds per decision)
- **Compounding API costs** on repetitive classifications
- **Fragility against rate limits and external service outages**

**`langgraph-jev-router`** solves this by inserting TypeSafe's Jev model as a **System 1 fast-path reflex router** in front of LangGraph conditional edges. When confident, Jev routes the request autonomously in **~100ms**; when uncertain, it falls back seamlessly to your existing LLM.

This repository benchmarks the performance, cost, and latency advantages of `langgraph-jev-router` across real industry datasets.

---

## 🏭 Industry Domains

| Domain | Status | Dataset | Key Result | Directory |
|---|---|---|---|---|
| **Manufacturing & Heavy Industry** | ✅ **Active** | [IHM Stefanini Industrial Safety Database](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) (CC0) | **-76% LLM calls**, **82ms latency** | [`manufacturing/`](./manufacturing/) |
| **Supply Chain & Logistics** | ✅ **Active** | [openFDA Food Enforcement Recall Database](https://api.fda.gov/food/enforcement.json) (Public Domain / 17 U.S.C. § 105) | **-72% LLM calls**, **88ms latency** | [`supply_chain/`](./supply_chain/) |
| **Healthcare & Clinical Triage** | ✅ **Active** | [MTSamples Clinical Transcription Database](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (CC0) | **-98% LLM calls**, **119ms latency** | [`healthcare/`](./healthcare/) |
| **Financial Services & Banking** | 🔜 *Roadmap* | Transaction fraud & dispute classification | *Coming soon* | `finance/` |
| **IT Service Desk & Customer Ops** | 🔜 *Roadmap* | Enterprise support ticket routing | *Coming soon* | `support/` |

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/JTankariya/jev-router-benchmark.git
cd jev-router-benchmark

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
TYPESAFE_API_KEY="your_typesafe_api_key"
GOOGLE_API_KEY="your_google_api_key"    # Optional: for live Gemini LLM baseline
```

### 3. Run a Benchmark

Run any industry benchmark:

```bash
# Manufacturing Domain Benchmark
python manufacturing/benchmark.py --sample 50
python manufacturing/benchmark.py --sweep

# Supply Chain & Logistics Domain Benchmark
python supply_chain/benchmark.py --sample 50
python supply_chain/benchmark.py --sweep

# Healthcare & Clinical Triage Benchmark
python healthcare/benchmark.py --sample 50
python healthcare/benchmark.py --sweep
```

---

## ⚖️ Dataset & Licensing Policy

All datasets used in this repository are verified for open licensing compliance:
- **Manufacturing:** Licensed under **CC0 1.0 Universal (Public Domain Dedication)**. See [`manufacturing/data/DATASET_LICENSE.md`](./manufacturing/data/DATASET_LICENSE.md) for full terms.
- Future industry datasets added to this repo will strictly adhere to permissive open licenses (CC0, MIT, Apache 2.0, or CC-BY).

---

## 🔗 Related Projects

- **[`langgraph-jev-router`](https://github.com/JTankariya/langgraph-jev-router)** — The core library enabling autonomous Jev fast-path routing in LangGraph.
- **[TypeSafe AI](https://typesafe.ai)** — System 1 model infrastructure and API.
