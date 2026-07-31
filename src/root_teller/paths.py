from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the configured data/artifact workspace.

    `ROOTTELLER_WORKSPACE` takes precedence. Falling back to the current
    directory keeps command-line use portable without embedding a machine path.
    """

    configured = os.environ.get("ROOTTELLER_WORKSPACE")
    root = Path(configured).expanduser() if configured else Path.cwd()
    return root.resolve()
