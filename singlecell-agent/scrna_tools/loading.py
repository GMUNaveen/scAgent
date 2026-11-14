"""Shared loader: 10X MTX -> QC filter -> normalize_total -> log1p -> attach labels.

This is the common foundation both the ML path and the NLP path build on (see the
`load_adata_obj_normal` stub in tool.py). Faithful to MyProject's notebooks, with one
deliberate upgrade: cell-type labels are joined to cells *by barcode* rather than by row
position. The notebooks assigned the annotation CSV to the filtered matrix positionally,
which only works when QC happens to reproduce exactly the annotated cell set. The
annotation CSV keys each cell as "{channel}_{barcode}" (e.g. "10X_P7_4_AAACCTGCACGACGAA"),
so we reconstruct that key and merge on it.
"""
from __future__ import annotations

import anndata as ad
import pandas as pd
import scanpy as sc

from . import config, workspace


def _read_sample(sample_dir) -> ad.AnnData:
    a = sc.read_10x_mtx(str(sample_dir))
    channel = config.channel_of(sample_dir)
    # Preserve provenance so labels can be joined by barcode later.
    a.obs["channel"] = channel
    # read_10x_mtx barcodes look like 'AAACCTGCACGACGAA-1'; annotations drop the '-N'.
    a.obs["barcode"] = [bc.rsplit("-", 1)[0] for bc in a.obs_names]
    return a


def _attach_labels(adata: ad.AnnData, tissues: list[str]) -> int:
    """Merge cell_ontology_class onto adata.obs by '{channel}_{barcode}', per tissue.

    Works uniformly for one or many tissues by joining within each tissue's rows.
    Returns the number of cells that received a label.
    """
    adata.obs["cell_ontology_class"] = pd.NA
    keys = adata.obs["channel"].astype(str) + "_" + adata.obs["barcode"].astype(str)
    for tissue in tissues:
        csv = config.annotation_csv_for(tissue)
        if csv is None:
            continue
        ann = pd.read_csv(csv).set_index("cell")["cell_ontology_class"]
        mask = (adata.obs["tissue"] == tissue).values
        adata.obs.loc[mask, "cell_ontology_class"] = keys[mask].map(ann).values
    return int(adata.obs["cell_ontology_class"].notna().sum())


def load_and_normalize(
    tissues: list[str] | str,
    min_genes: int = 500,
    min_counts: int = 1000,
    target_sum: float = 1e4,
    max_cells: int | None = None,
    attach_labels: bool = True,
    drop_unlabeled: bool = False,
    persist: bool = False,
) -> dict:
    """Load one or more tissues, QC-filter, normalize_total + log1p, attach labels.

    Returns a summary dict including the workspace 'handle' for downstream tools.
    """
    if isinstance(tissues, str):
        tissues = [tissues]

    parts: list[ad.AnnData] = []
    resolved: dict[str, int] = {}
    for tissue in tissues:
        dirs = config.sample_dirs_for(tissue)
        if not dirs:
            raise FileNotFoundError(
                f"No 10X sample directories for tissue '{tissue}' under {config.DATA_ROOT}. "
                f"Available: {config.available_tissues()}"
            )
        for d in dirs:
            a = _read_sample(d)
            a.obs["tissue"] = tissue
            parts.append(a)
        resolved[tissue] = len(dirs)

    adata = parts[0] if len(parts) == 1 else ad.concat(parts, join="outer")
    adata.obs_names_make_unique()
    n_raw = adata.n_obs

    # QC filter (mirrors MyProject: min_genes then min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    n_qc = adata.n_obs

    if max_cells is not None and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0)

    # Normalize + log1p (the shared base; scaling is ML-path specific)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    n_labeled = 0
    if attach_labels:
        n_labeled = _attach_labels(adata, tissues)
        if drop_unlabeled:
            adata = adata[adata.obs["cell_ontology_class"].notna()].copy()

    meta = {
        "stage": "normalized",
        "tissues": tissues,
        "samples_per_tissue": resolved,
        "n_raw": n_raw,
        "n_after_qc": n_qc,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_labeled": int(n_labeled),
    }
    handle = workspace.put(adata, meta=meta, persist=persist)
    return {"handle": handle, **meta}


def list_available_tissues() -> dict:
    tissues = config.available_tissues()
    return {
        "data_root": str(config.DATA_ROOT),
        "n_tissues": len(tissues),
        "tissues": tissues,
    }
