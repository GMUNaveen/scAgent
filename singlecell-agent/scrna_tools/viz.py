"""Visualization: t-SNE plots colored by true / predicted cell type (from MyProject notebooks).

Mirrors the notebook's neighbors -> t-SNE -> sc.pl.tsne steps (cells 21-42), but renders to
PNG files (headless Agg backend) and returns their paths, since a tool can't pop a window.
Run after train_ml_models (with store_predictions=True) so the prediction fields exist.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt  # noqa: E402
import scanpy as sc  # noqa: E402

from . import config, workspace  # noqa: E402

# Human-readable titles for the fields we know how to color by.
_TITLES = {
    "cell_ontology_class": "True Cell Types",
    "ml_pred_label": "Predicted (ML)",
    "ml_correct": "Correct vs Incorrect",
    "louvain": "Louvain Clusters",
}


def _ensure_tsne(adata, n_neighbors: int, n_pcs: int, perplexity: float) -> None:
    if "X_pca" not in adata.obsm:
        raise ValueError(
            "Artifact has no PCA (obsm['X_pca']). Run preprocess_ml before visualizing."
        )
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    if "X_tsne" not in adata.obsm:
        sc.tl.tsne(adata, n_pcs=n_pcs, perplexity=perplexity, random_state=0)


def plot_tsne(
    handle: str,
    color_by: list[str] | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 10,
    perplexity: float = 30,
) -> dict:
    """Compute t-SNE (if needed) and save one PNG per requested coloring.

    color_by defaults to whichever of the true/predicted/correctness fields are present.
    Returns {'handle', 'figures': [paths], 'colored_by': [fields]}.
    """
    adata = workspace.get(handle)
    _ensure_tsne(adata, n_neighbors, n_pcs, perplexity)

    if color_by is None:
        color_by = [f for f in ("cell_ontology_class", "ml_pred_label", "ml_correct") if f in adata.obs]
    if not color_by:
        raise ValueError(
            "Nothing to color by. Load with labels and run train_ml_models first, "
            "or pass color_by explicitly."
        )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths, used = [], []
    for field in color_by:
        if field not in adata.obs:
            continue
        ax = sc.pl.tsne(
            adata, color=field, title=_TITLES.get(field, field), show=False, legend_loc="on data"
        )
        fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
        out = config.FIGURES_DIR / f"tsne_{field}_{handle}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out))
        used.append(field)

    return {"handle": handle, "figures": paths, "colored_by": used}
