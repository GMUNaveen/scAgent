"""In-process artifact registry so tools can pass AnnData objects by *handle*.

Tools return short strings to the LLM, not multi-gigabyte matrices. So each tool stores
its AnnData in this registry under a handle like 'adata_1' and returns that handle plus a
human-readable summary. The next tool receives the handle and looks the object back up.

The registry is a module-level singleton: the whole LangGraph agent runs in one process,
so every tool call shares it. Objects are also optionally mirrored to disk as .h5ad for
inspection / crash recovery.
"""
from __future__ import annotations

from typing import Any

from . import config

# handle -> {"adata": AnnData, "meta": {...}}
_REGISTRY: dict[str, dict[str, Any]] = {}
_COUNTER = {"n": 0}


def _next_handle(prefix: str = "adata") -> str:
    _COUNTER["n"] += 1
    return f"{prefix}_{_COUNTER['n']}"


def put(adata, meta: dict | None = None, prefix: str = "adata", persist: bool = False) -> str:
    """Register an AnnData and return its handle."""
    handle = _next_handle(prefix)
    _REGISTRY[handle] = {"adata": adata, "meta": dict(meta or {})}
    if persist:
        save_to_disk(handle)
    return handle


def get(handle: str):
    """Return the AnnData for a handle, or raise a clear error the agent can relay."""
    if handle not in _REGISTRY:
        raise KeyError(
            f"No artifact named '{handle}'. Known handles: {list(_REGISTRY) or '(none)'}. "
            f"Call a loading tool first to create one."
        )
    return _REGISTRY[handle]["adata"]


def get_meta(handle: str) -> dict:
    if handle not in _REGISTRY:
        raise KeyError(f"No artifact named '{handle}'.")
    return _REGISTRY[handle]["meta"]


def update_meta(handle: str, **kw) -> None:
    get_meta(handle).update(kw)


def list_handles() -> list[dict]:
    out = []
    for h, entry in _REGISTRY.items():
        a = entry["adata"]
        out.append({"handle": h, "n_obs": int(a.n_obs), "n_vars": int(a.n_vars), **entry["meta"]})
    return out


def save_to_disk(handle: str) -> str:
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.WORKSPACE_DIR / f"{handle}.h5ad"
    get(handle).write_h5ad(path)
    return str(path)
