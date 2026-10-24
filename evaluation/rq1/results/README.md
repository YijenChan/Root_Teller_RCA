# RQ1 result files

- `paper_table.csv` transcribes the supplied revised manuscript's nine-method
  RQ1 table (27 dataset/method rows).
- `paper_table_provenance.json` records its transcription origin and explicitly
  states that no new evaluation was run to generate it.
- `paper_table_legacy_r1.csv` preserves the earlier seven-method transcription;
  it must not be mixed into current-method aggregation.
- It is a compact paper-facing summary, not an inference input or an execution
  cache.
- The public package does not redistribute experiment-host API responses,
  private labels, or per-case predictions.
- Independent reruns can export seed-level summaries with
  `../scripts/aggregate_seed_results.py` after completing the documented
  protocol locally.

Model/backbone parity, fine-interval weights, and execution of revised
cross-window/refinement mechanisms are separate provenance questions. See
[`docs/REVISION2_ALIGNMENT.md`](../../../docs/REVISION2_ALIGNMENT.md).
