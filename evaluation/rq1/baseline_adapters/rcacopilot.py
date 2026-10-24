"""Portable entry point for the documented RCACopilot benchmark adaptation."""
import sys
from root_teller.baselines.run import main

if __name__ == "__main__":
    sys.argv.extend(["--method", "rcacopilot"])
    main()
