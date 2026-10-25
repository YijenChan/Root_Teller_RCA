# RCACopilot benchmark adaptation

Entry point: `python -m root_teller.baselines.run --method rcacopilot --help`.
Implementation: `src/root_teller/baselines/methods.py` (`Copilot`).

This is a paper-based reproduction, not official author code or an endorsement
of a third-party repository. It retains FastText retrieval, Euclidean-distance
similarity with temporal decay, five distinct historical categories,
120-140-word summaries and example selection. Categories are adapted to service
identities. A selected known category fixes Top-1; the model orders remaining
services. Unknown/duplicate/out-of-roster outputs are rejected, never completed
using private labels or anomaly-score fallback.

Historical examples use the inner training set for validation and the entire
outer development set for outer test. The runner records manifest hashes,
model, sampling parameters, source hashes, call/token budgets and failures.
