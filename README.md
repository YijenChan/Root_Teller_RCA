# Agentic RCA: Multi-Modal Root Cause Analysis with Agentic AI

> 🚀 **Agentic RCA** is an experimental pipeline for cybersecurity root cause analysis (RCA),
> combining **multi-modal perception**, **graph reasoning (R-GAT+time)**,
> and **LLM-based reporting with agentic feedback & verification**.

---

## ✨ Features

* **Stage A – Perception**
  Mid-level fusion of **logs / metrics / traces** with adaptive quality-aware gating.

* **Stage B – Reasoning**
  **R-GAT(+time)** graph neural network for root-cause localization & attack-chain reconstruction.

* **Stage C – Reporter**
  LLM-enhanced Markdown reports with **uncertainty margins, modal contributions, and remediation advice**.
  Fallback templates ensure robustness when LLM is unavailable.

* **Verifier Agent**
  Independent consistency check between evidence constraints and LLM predictions.

* **Feedback Agent**
  Human-in-the-loop reinforcement: stores structured feedback and updates a persistent `profile.yaml`.

---

## 🧩 Methodology Overview

（此处留空，方便你贴 methodology 图，例如整体架构或流程图）

---

## 📂 Project Structure

```text
agentic_rca/
├── agents/              # Agent implementations (perception, reasoner, reporter, verifier, feedback, etc.)
├── configs/             # YAML configs for runs and paths
├── data/                # Single-case demo builder
├── feedback/            # Feedback schema, prompts, reward model
├── pipelines/           # End-to-end pipelines (single_case, agentic_agents, validation, feedback demo)
├── stage_a_perception/  # Fusion encoders & adapters
├── stage_b_reasoning/   # R-GAT(+time) model & training
├── stage_c_reporter/    # LLM client & predictor agent
├── tools/               # Collectors for logs/metrics/traces (simulated)
├── utils/               # IO, env, viz, seed utils
└── outputs/             # Generated artifacts (perception / reasoning / reports / feedback memory)
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run single-case pipeline

```bash
python -m pipelines.run_single_case
```

This will:

* Encode a synthetic **demo case**
* Train R-GAT(+time)
* Save outputs under `outputs/`

### 3. Run full agentic pipeline

```bash
python -m pipelines.run_agentic_agents
```

This will:

* Launch an **EventBus**
* Sequentially run **Stage A → Stage B → Stage C → Verifier**
* Save artifacts and reports under `outputs/`

---

## 📊 Sample Output

* **Perception**: `outputs/perception/mm_node_embeddings.pt`
* **Reasoning**:

  * model → `outputs/reasoning/rgat_time_min.pt`
  * metrics → `outputs/reasoning/single_case.json`
* **Reports**:

  * text summary → `outputs/reports/single_case_report.txt`
  * LLM Markdown → `outputs/reports/llm_report.md`
  * raw JSON → `outputs/reports/llm_report_raw.json`

*(示例报告截图或片段可以放在这里)*

---

## 🔍 Verification & Feedback

* **VerifierAgent** checks consistency of predicted chains vs. evidence constraints.
* **FeedbackAgent** allows storing structured feedback (`feedback_store.jsonl`) and updating memory (`profile.yaml`).

Run a demo feedback loop:

```bash
python -m pipelines.demo_feedback_loop
```

---

## 🛠 Roadmap

* [ ] Extend Stage-A encoders (support additional modalities)
* [ ] Plug-in alternative GNN baselines (GCN, R-GCN, TGAT)
* [ ] Expand LLM reporter with adaptive prompt rules (`feedback/prompts/prompt_rules.yaml`)
* [ ] Integrate reinforcement signals from feedback into training loop

---

## 📜 License

MIT License.
Research prototype — not production-ready.

---

## 🙌 Acknowledgements

* Inspired by research on **AISecOps**, **multi-modal RCA**, and **agentic AI frameworks**.
* Thanks to open-source GNN/LLM ecosystems and feedback contributors.

