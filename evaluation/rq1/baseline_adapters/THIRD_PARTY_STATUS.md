# Third-party source status

The compatibility adapters in this directory are Root-Teller artifact code.
The diagnostic implementations under the repository-level `baselines/`
directory originate from their cited upstream projects and retain their
upstream terms.

License files are present in the staged copies of Multi-source RCD, Nezha,
TORAI, ThinkFL, and RCLAgent. No license file was found in the inspected Eadro
upstream snapshot, so the public package retains only its provenance URL,
acquisition instructions, and the Root-Teller adapter. Obtain Eadro itself
from its official repository unless the repository owner later confirms
separate redistribution permission.

This notice is factual inventory information, not legal advice and not a
replacement for a repository-level license selected by the Root-Teller
authors.

RCACopilot and RCAgent are additional Root-Teller-authored, paper-based
adaptations in `src/root_teller/baselines/`, not bundled official author source.
Run these adaptations with `python -m root_teller.baselines.run`, using explicit
model and sampling settings.
