"""Paths and dataset discovery for the Tabula Muris 10X droplet data.

There is a single shared copy of the data in <project>/data/ (not tracked by git; fetch it
with scripts/get_data.sh). MyProject/ and NLPFinalProjectCode/ symlink to it. Override with
the SCRNA_DATA_ROOT / SCRNA_ANNOTATIONS_DIR env vars to point elsewhere.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# .../singlecell-agent/scrna_tools/config.py -> project root is two parents up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("SCRNA_DATA_ROOT", PROJECT_ROOT / "data" / "droplet"))
ANNOTATIONS_DIR = Path(
    os.environ.get("SCRNA_ANNOTATIONS_DIR", PROJECT_ROOT / "data" / "annotations")
)

# Where tools may persist AnnData artifacts (.h5ad) so a run can be inspected/resumed.
WORKSPACE_DIR = Path(os.environ.get("SCRNA_WORKSPACE_DIR", PROJECT_ROOT / ".scrna_workspace"))

# Where visualization tools write PNG figures.
FIGURES_DIR = Path(os.environ.get("SCRNA_FIGURES_DIR", PROJECT_ROOT / ".scrna_figures"))

# A 10X sample directory looks like "Heart-10X_P7_4". The channel is everything after
# the first hyphen ("10X_P7_4"); annotation CSVs key cells as "{channel}_{barcode}".
_SAMPLE_RE = re.compile(r"^(?P<tissue>[A-Za-z_]+)-(?P<channel>10X_.+)$")


def sample_dirs_for(tissue: str) -> list[Path]:
    """All 10X sample directories belonging to a tissue, e.g. tissue='Heart'."""
    if not DATA_ROOT.exists():
        return []
    hits = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue
        m = _SAMPLE_RE.match(p.name)
        if m and m.group("tissue").lower() == tissue.lower():
            hits.append(p)
    return hits


def channel_of(sample_dir: Path) -> str:
    """'Heart-10X_P7_4' -> '10X_P7_4' (the channel id used in annotation CSVs)."""
    m = _SAMPLE_RE.match(sample_dir.name)
    return m.group("channel") if m else sample_dir.name


def annotation_csv_for(tissue: str) -> Path | None:
    """Path to '{Tissue}_droplet_annotation.csv' if it exists."""
    cand = ANNOTATIONS_DIR / f"{tissue}_droplet_annotation.csv"
    return cand if cand.exists() else None


def available_tissues() -> list[str]:
    """Distinct tissue prefixes discoverable under DATA_ROOT."""
    seen: dict[str, None] = {}
    if DATA_ROOT.exists():
        for p in sorted(DATA_ROOT.iterdir()):
            m = _SAMPLE_RE.match(p.name) if p.is_dir() else None
            if m:
                seen.setdefault(m.group("tissue"), None)
    return list(seen)
