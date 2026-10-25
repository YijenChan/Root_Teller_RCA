# Baseline provenance and source snapshots

This directory records eight baseline entries. The six original provenance
entries are retained, and two additional paper-based adaptations are documented. Where redistribution
terms are available, a convenience source snapshot is included. Portable
Root-Teller-specific adapters are provided separately under
`evaluation/rq1/baseline_adapters/`. Generated predictions, caches, benchmark
data, and private reproduction outputs are intentionally excluded.

| Directory | Method | Upstream artifact |
|---|---|---|
| `eadro` | Eadro | <https://github.com/BEbillionaireUSD/Eadro> |
| `nezha` | Nezha | <https://github.com/IntelligentDDS/Nezha> |
| `multisource_rcd` | Multi-source RCD | <https://github.com/phamquiluan/RCAEval> |
| `torai` | TORAI | <https://figshare.com/articles/software/31938495> |
| `thinkfl` | ThinkFL | [Training](https://github.com/LLM4AIOps/OpenRLHF-ThinkFL), [inference](https://github.com/LLM4AIOps/ThinkFL), [weights](https://huggingface.co/Zhang-Lingzhe/ThinkFL-Llama3.2-3B) |
| `rclagent` | RCLAgent | <https://github.com/LLM4AIOps/RCLAgent-V2> |
| `rcacopilot` | RCACopilot adaptation | Root-Teller paper-based implementation; see directory README |
| `rcagent` | RCAgent CIKM 2024 adaptation | Root-Teller paper-based implementation; see directory README |

Run the added adaptations with `python -m root_teller.baselines.run`, selecting
the model, sampling parameters, and inference budget explicitly.

Each baseline remains subject to its upstream license and dependency terms.
The Eadro directory contains only provenance and acquisition instructions
because no license file was found in the inspected upstream snapshot; obtain
its implementation directly from the official repository. Included snapshots
are conveniences rather than authoritative upstream distributions.
