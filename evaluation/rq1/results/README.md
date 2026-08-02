# RQ1 result files

- `paper_table.csv` is the frozen table reported in the manuscript.
- It is a compact paper-facing summary, not an inference input or an execution
  cache.
- The public package does not redistribute experiment-host API responses,
  private labels, or per-case predictions.
- Independent reruns can export seed-level summaries with
  `../scripts/aggregate_seed_results.py` after completing the documented
  protocol locally.
