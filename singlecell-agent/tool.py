from langchain.tools import tool
from langchain_groq import ChatGroq

from scrna_tools import loading, ml_annotate, nlp_annotate, viz, compare


model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ─────────────────────────────── scRNA annotation tools ───────────────────────────────
# These wrap the extracted pipeline modules. Each returns a short text summary for the LLM;
# heavy AnnData objects stay in the workspace registry and are passed between tools by
# their string `handle`.

@tool
def list_available_tissues() -> str:
    """List the tissues available in the local Tabula Muris 10X dataset.

    Use this first when the user is unsure which tissue to annotate.
    """
    info = loading.list_available_tissues()
    return f"{info['n_tissues']} tissues under {info['data_root']}:\n" + ", ".join(info["tissues"])


@tool
def load_and_normalize(tissue: str, max_cells: int | None = None) -> str:
    """Load a tissue's 10X data, QC-filter, total-normalize and log1p, and attach cell-type labels.

    This is the shared first step for BOTH the ML and NLP annotation paths. Returns a
    `handle` (e.g. 'adata_1') that later tools need.

    Args:
        tissue: Tissue name, e.g. 'Heart', 'Liver', 'Kidney'. Use list_available_tissues if unsure.
        max_cells: Optional cap on number of cells (subsample) for a fast run. None uses all cells.
    """
    r = loading.load_and_normalize(tissue, max_cells=max_cells)
    return (
        f"Loaded {tissue}: {r['n_raw']} raw cells -> {r['n_after_qc']} after QC -> "
        f"{r['n_cells']} used, {r['n_genes']} genes, {r['n_labeled']} labeled.\n"
        f"handle = {r['handle']}"
    )


@tool
def preprocess_ml(handle: str) -> str:
    """Prepare a normalized dataset for ML annotation: scale, select highly-variable genes, run PCA.

    Run this after load_and_normalize and before train_ml_models.

    Args:
        handle: The handle returned by load_and_normalize.
    """
    r = ml_annotate.preprocess_ml(handle)
    return (
        f"ML preprocess of {handle}: {r['n_hvg']} highly-variable genes kept, "
        f"PCA with {r['n_pcs']} components, {r['n_cells']} cells.\n"
        f"handle = {r['handle']}"
    )


@tool
def train_ml_models(handle: str, models: list[str] | None = None) -> str:
    """Train ML classifiers to annotate cell types and report test accuracy for each.

    Run this after preprocess_ml. Available models: logistic_regression, svm, random_forest,
    gradient_boosting, knn, naive_bayes, neural_network (a PyTorch MLP). Pass a subset to run
    only those, or omit for all.

    Args:
        handle: The handle returned by preprocess_ml.
        models: Optional list of model names to train. None trains the full menu.
    """
    r = ml_annotate.train_ml_models(handle, models=models)
    lines = [
        f"Trained on {r['n_train']} cells, tested on {r['n_test']}, {r['n_classes']} cell types.",
        "Accuracy by model:",
    ]
    for name, res in sorted(r["results"].items(), key=lambda kv: kv[1]["accuracy"], reverse=True):
        lines.append(f"  {name}: acc={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    lines.append(f"Best: {r['best_model']} ({r['best_accuracy']:.4f})")
    return "\n".join(lines)


@tool
def visualize_tsne(handle: str, color_by: list[str] | None = None) -> str:
    """Render t-SNE plots of the cells, saved as PNG files, colored by cell type / prediction.

    Run after train_ml_models. By default it produces three plots: true cell types,
    ML-predicted labels, and correct-vs-incorrect. Returns the saved image file paths.

    Args:
        handle: The ML-preprocessed handle that was passed to train_ml_models.
        color_by: Optional list of obs fields to color by (e.g. ['cell_ontology_class',
            'ml_pred_label', 'ml_correct']). None uses all available.
    """
    r = viz.plot_tsne(handle, color_by=color_by)
    lines = [f"Saved {len(r['figures'])} t-SNE figure(s) for {handle}:"]
    for field, path in zip(r["colored_by"], r["figures"]):
        lines.append(f"  {field}: {path}")
    return "\n".join(lines)


# ─────────────────────────────── NLP annotation tools ───────────────────────────────
# These branch off the SAME load_and_normalize handle as the ML path.

@tool
def build_gene_rank_tokens(handle: str, top_k: int = 100) -> str:
    """Turn each labeled cell into a 'sentence' of its top-K highest-expressed genes (NLP framing).

    Run after load_and_normalize; prepares data for the BiLSTM. Uses the labels the loader
    already attached. Run this before train_bilstm / run_order_experiment.

    Args:
        handle: The handle returned by load_and_normalize.
        top_k: Number of top-expressed genes to keep per cell (sequence length). Default 100.
    """
    r = nlp_annotate.build_gene_rank_tokens(handle, top_k=top_k)
    common = ", ".join(f"{lab} ({n})" for lab, n in r["top_common_labels"])
    return (
        f"Built {r['n_sentences']} gene-rank sentences from {handle}: "
        f"vocab={r['vocab_size']}, {r['num_classes']} cell types.\n"
        f"Most common: {common}\nhandle = {handle}"
    )


@tool
def train_bilstm(handle: str, epochs: int = 5) -> str:
    """Train a BiLSTM to annotate cell types from the gene-rank token sentences.

    Run after build_gene_rank_tokens. Reports test accuracy and macro-F1.

    Args:
        handle: The handle prepared by build_gene_rank_tokens.
        epochs: Training epochs. Default 5.
    """
    r = nlp_annotate.train_bilstm(handle, epochs=epochs)
    return (
        f"BiLSTM trained ({r['epochs']} epochs, {r['num_classes']} cell types) on {handle}:\n"
        f"  test accuracy = {r['test_accuracy']:.4f}\n"
        f"  test macro-F1 = {r['test_macro_f1']:.4f}"
    )


@tool
def run_order_experiment(handle: str, epochs: int = 5) -> str:
    """Test whether gene *order* matters: train the BiLSTM on ordered vs. shuffled sequences.

    Run after build_gene_rank_tokens. Trains twice (original ranking vs. randomized order,
    same gene set) and reports the performance gap, plus a comparison bar chart.

    Args:
        handle: The handle prepared by build_gene_rank_tokens.
        epochs: Training epochs per run. Default 5.
    """
    r = nlp_annotate.run_order_experiment(handle, epochs=epochs)
    return (
        f"Order-sensitivity experiment on {handle} ({r['epochs']} epochs each):\n"
        f"  Ordered:  acc={r['ordered']['accuracy']:.4f}  macro-F1={r['ordered']['macro_f1']:.4f}\n"
        f"  Shuffled: acc={r['shuffled']['accuracy']:.4f}  macro-F1={r['shuffled']['macro_f1']:.4f}\n"
        f"  Accuracy drop from shuffling = {r['accuracy_drop']:.4f}\n"
        f"  Chart: {r['figure']}"
    )


# ─────────────────────────────── compare both approaches ───────────────────────────────

@tool
def compare_approaches(handle: str, nlp_epochs: int = 5) -> str:
    """Run BOTH the ML and NLP annotation approaches on one loaded dataset and compare them.

    Run after load_and_normalize. Internally runs the full ML path (preprocess + model menu)
    and the full NLP path (gene-rank tokens + BiLSTM) on the same cells, then reports which
    achieved higher accuracy.

    Args:
        handle: The handle returned by load_and_normalize.
        nlp_epochs: Epochs for the BiLSTM. Default 5.
    """
    r = compare.run_comparison(handle, nlp_epochs=nlp_epochs)
    return (
        f"Comparison on {r['tissues']} ({r['n_cells']} cells):\n"
        f"  ML  ({r['ml']['best_model']}): acc={r['ml']['accuracy']:.4f}  "
        f"macro-F1={r['ml']['macro_f1']:.4f}\n"
        f"  NLP (bilstm): acc={r['nlp']['accuracy']:.4f}  macro-F1={r['nlp']['macro_f1']:.4f}\n"
        f"  Higher accuracy: {r['winner_by_accuracy']}"
    )


# ─────────────────────────────── arithmetic demo tools ───────────────────────────────
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    if b == 0:
        return float("nan")
    return a / b


# Augment the LLM with tools
tools = [
    list_available_tissues,
    load_and_normalize,
    preprocess_ml,
    train_ml_models,
    visualize_tsne,
    build_gene_rank_tokens,
    train_bilstm,
    run_order_experiment,
    compare_approaches,
    add,
    multiply,
    divide,
]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
