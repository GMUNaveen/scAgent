"""Run BOTH annotation approaches on one loaded dataset and tabulate the results.

Given a handle from load_and_normalize, this runs the ML path (preprocess -> train) and the
NLP path (gene-rank tokens -> BiLSTM) on the same cells, so the two methods can be compared
head-to-head. The paths use separate derived data (ML scales + reduces to HVG/PCA; NLP works
off the normalized counts), so they don't interfere.
"""
from __future__ import annotations

from . import ml_annotate, nlp_annotate, workspace


def run_comparison(
    handle: str,
    ml_models: list[str] | None = None,
    nlp_epochs: int = 5,
) -> dict:
    """Run ML and NLP annotation on `handle` and return both plus a comparison summary."""
    base = workspace.get_meta(handle)

    # ML path (creates its own derived handle for the HVG/PCA matrix)
    ml_handle = ml_annotate.preprocess_ml(handle)["handle"]
    ml = ml_annotate.train_ml_models(ml_handle, models=ml_models)
    ml_best = ml["best_model"]
    ml_summary = {
        "best_model": ml_best,
        "accuracy": ml["results"][ml_best]["accuracy"],
        "macro_f1": ml["results"][ml_best]["macro_f1"],
        "all_models": {m: r["accuracy"] for m, r in ml["results"].items()},
    }

    # NLP path (operates on the original normalized handle)
    nlp_annotate.build_gene_rank_tokens(handle)
    nlp = nlp_annotate.train_bilstm(handle, epochs=nlp_epochs)
    nlp_summary = {
        "model": "bilstm",
        "accuracy": nlp["test_accuracy"],
        "macro_f1": nlp["test_macro_f1"],
    }

    winner = "ML" if ml_summary["accuracy"] >= nlp_summary["accuracy"] else "NLP"
    return {
        "handle": handle,
        "tissues": base.get("tissues"),
        "n_cells": base.get("n_cells"),
        "ml": ml_summary,
        "nlp": nlp_summary,
        "winner_by_accuracy": winner,
    }
