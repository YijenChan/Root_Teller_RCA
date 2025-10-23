# 🧠 Agentic RCA v1.2

### Multi-Modal Root Cause Analysis with Agentic AI

> 🚀 **Agentic RCA** is a reproducible framework for cybersecurity root-cause analysis (RCA),
> integrating **multi-modal perception**, **graph reasoning (R-GAT + time)**,
> and **LLM-based reporting with agentic feedback and verification**.

---

## ✨ Key Features

| Stage                 | Module                                  | Purpose                                                                              |
| :-------------------- | :-------------------------------------- | :----------------------------------------------------------------------------------- |
| 🅰 **Perception**     | `stage_a_perception/`                   | Mid-level fusion of **logs / metrics / traces** using adaptive quality-aware gating. |
| 🅱 **Reasoning**      | `stage_b_reasoning/`                    | **R-GAT (+time)** network for root-cause localization & attack-chain reconstruction. |
| 🅲 **Reporter**       | `stage_c_reporter/`                     | LLM-generated Markdown reports with confidence margins & modal contributions.        |
| 🔎 **Verifier Agent** | `stage_c_reporter/feedback_profiler.py` | Consistency check between evidence and LLM predictions.                              |
| 🔁 **Feedback Agent** | `feedback/`                             | Human-in-the-loop reinforcement updating a persistent profile store.                 |

---

## 🧩 Methodology Overview

The **Agentic RCA Pipeline** follows a three-agent architecture:

```
Perception Agent  →  Reasoner Agent (R-GAT + time)  →  Reporter Agent (LLM + Verifier)
          ↑                                                ↓
          └───────────────  Feedback Loop (Consistency + Human Review) ───────────────┘
```

Each agent communicates via an **EventBus**, ensuring asynchronous yet verifiable execution.
Outputs include both numerical metrics and explainable RCA reports.

<img width="1269" height="738" alt="{7EF0E8FA-AC5D-4BB7-AF41-89C4ABA6DF92}" src="https://github.com/user-attachments/assets/e515f19a-dfe5-446b-ad40-3a430dedd9c0" />

---

## 📂 Project Structure

```text
agentic_rca/
├── agents/              # Agentic wrappers (Perception / Reasoner / Reporter / Verifier / Feedback)
├── configs/             # YAML configs (run paths, LLM endpoints)
├── feedback/            # Feedback schema & reward model
├── pipelines/           # End-to-end pipelines (single_case, agentic_agents)
├── stage_a_perception/  # Multi-modal fusion (logs / metrics / traces)
├── stage_b_reasoning/   # R-GAT(+time) model & training scripts
├── stage_c_reporter/    # LLM client, predictor, verifier, feedback profiler
├── utils/               # IO / env / seed helpers
└── outputs/             # Generated artifacts (perception / reasoning / reports / feedback)
```
<img width="970" height="626" alt="{7AE8656B-9783-40DD-8BAA-A7E5D46EAF61}" src="https://github.com/user-attachments/assets/572690fb-3ba0-4c2c-a557-bfc93a4b7ed6" />

---

## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run a Single-Case Pipeline

```bash
python -m pipelines.run_single_case
```

This will:

* Encode a synthetic demo case
* Train R-GAT(+time)
* Generate LLM report and feedback stats under `outputs/`

### 3️⃣ Optional — Run Full Agentic Pipeline

```bash
python -m pipelines.run_agentic_agents
```

This launches an **EventBus** and executes
`Stage A → Stage B → Stage C → Verifier`.

---

## 📊 Sample Outputs

| Category       | Artifact Path                              | Description                                  |
| -------------- | ------------------------------------------ | -------------------------------------------- |
| **Perception** | `outputs/perception/mm_node_embeddings.pt` | Node embeddings and modal weights (α)        |
| **Reasoning**  | `outputs/reasoning/single_case.json`       | Evaluation metrics + root/chain predictions  |
| **Reporter**   | `outputs/reports/llm_report.md`            | LLM-generated RCA Markdown report            |
| **Verifier**   | `outputs/reports/feedback_stats.json`      | Consistency / Jaccard / pass rate statistics |
| **Event Log**  | `outputs/event_log.jsonl`                  | All agent messages via EventBus              |

<img width="1040" height="783" alt="{55C0092F-CB77-458E-B7F4-CD44CAA1C681}" src="https://github.com/user-attachments/assets/2c5b8be2-f7ee-44f1-8583-da1defa03a51" />

---

## 🔍 Verification & Feedback Loop

* **VerifierAgent** cross-checks predicted root/chain against evidence constraints.
* **FeedbackAgent** stores human feedback (`feedback_store.jsonl`) and updates persistent `profile.yaml`.

Run a demo feedback loop:

```bash
python -m pipelines.demo_feedback_loop
```

---

## 🧪 Reproducibility Checklist

| Item                                      | Status |
| :---------------------------------------- | :----- |
| GPU-based training (R-GAT)                | ✅      |
| Deterministic seed control                | ✅      |
| YAML configurable paths and LLM endpoints | ✅      |
| Fallback template when LLM disabled       | ✅      |
| Outputs reproducible under `outputs/`     | ✅      |

---

## 🛠 Roadmap

* [ ] Expand Stage-A encoders for richer modalities
* [ ] Integrate feedback-driven reinforcement into Reasoner
* [ ] Support multi-agent parallel execution (A2A protocol)
* [ ] Add visual RCA graph and interactive report viewer

---





