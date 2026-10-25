from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from root_teller.baselines.classical_run import main as run


def main():
    arguments = sys.argv[1:]
    if "--dataset" not in arguments:
        arguments = ["--dataset", "re2_ob", *arguments]
    sys.argv = [sys.argv[0], "--method", "torai", *arguments]
    run()


if __name__ == "__main__":
    main()
