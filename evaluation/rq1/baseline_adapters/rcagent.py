"""Portable entry point for the documented CIKM 2024 RCAgent adaptation."""
import sys
from root_teller.baselines.run import main

if __name__ == "__main__":
    sys.argv.extend(["--method", "rcagent"])
    main()
