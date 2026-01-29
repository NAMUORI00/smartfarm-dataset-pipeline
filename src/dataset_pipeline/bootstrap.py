from __future__ import annotations

import os
import sys
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def resolve_workspace_root() -> Path | None:
    """
    Resolve workspace root directory.

    Priority:
      1) SMARTFARM_WORKSPACE_ROOT
      2) Assume this repo is a submodule of the workspace.
    """
    from_env = _env_path("SMARTFARM_WORKSPACE_ROOT")
    if from_env is not None:
        return from_env

    # .../<workspace>/smartfarm-ingest/src/dataset_pipeline/bootstrap.py
    #              ^ parents[3]
    candidate = Path(__file__).resolve().parents[3]
    return candidate if candidate.exists() else None


def resolve_search_root(workspace_root: Path | None) -> Path | None:
    """
    Resolve smartfarm-search repository root.

    Priority:
      1) SMARTFARM_SEARCH_ROOT
      2) <workspace>/smartfarm-search (when workspace_root is available)
    """
    from_env = _env_path("SMARTFARM_SEARCH_ROOT")
    if from_env is not None:
        return from_env

    if workspace_root is None:
        return None

    candidate = (workspace_root / "smartfarm-search").resolve()
    return candidate if candidate.exists() else None


def ensure_search_on_path() -> Path:
    """Ensure smartfarm-search is importable (for `import core ...`)."""
    workspace_root = resolve_workspace_root()
    search_root = resolve_search_root(workspace_root)
    if search_root is None:
        raise RuntimeError(
            "Cannot locate smartfarm-search. Set SMARTFARM_SEARCH_ROOT "
            "(or SMARTFARM_WORKSPACE_ROOT)."
        )

    search_str = str(search_root)
    if search_str not in sys.path:
        sys.path.insert(0, search_str)

    return search_root

