# singlecell-agent

A conversational agent that annotates single-cell RNA-seq (scRNA-seq) cell types by
running the code from two projects behind one natural-language interface:

- **MyProject** — a *Machine Learning* approach (per-tissue supervised classifiers)
- **NLPFinalProjectCode** — an *NLP* approach (each cell → a "sentence" of its top-expressed
  genes, classified by a BiLSTM), plus an order-sensitivity experiment

Both operate on the same Tabula Muris 10X droplet data and share one preprocessing base, so
you can pick either approach — or compare them — from a single chat request.

## Architecture

```
build_compile.py      LangGraph agent: START → llm_call ⇄ tool_node → END
├── model_node.py      llm_call: the LLM (Groq qwen) decides which tool to call
├── tool_node.py       executes the chosen tool
├── end_logic.py       loop until the LLM stops calling tools
├── state.py           message state
└── tool.py            @tool definitions (thin wrappers) + model.bind_tools(...)
        │
        └── scrna_tools/         the extracted pipeline code (importable, tested)
            ├── config.py        data paths + tissue/sample discovery
            ├── workspace.py     handle registry (pass AnnData between tools by id)
            ├── loading.py       SHARED: load 10X → QC → normalize → log1p → labels
            ├── ml_annotate.py   ML path: scale/HVG/PCA + model menu (incl. PyTorch MLP)
            ├── nlp_annotate.py  NLP path: gene-rank tokens → BiLSTM → order experiment
            ├── viz.py           t-SNE plots (true vs predicted)
            └── compare.py       run both approaches and tabulate
```

### Why handles?

Tools return short text to the LLM, not multi-GB matrices. Each tool stores its `AnnData` in
`workspace` under a handle (e.g. `adata_1`) and returns that handle plus a summary. The next
tool receives the handle and looks the object back up. This is how state flows through the chain.

## Tool chains

Every path starts with `load_and_normalize`, which returns the shared handle.

**ML annotation**
```
load_and_normalize("Heart")      → adata_1
preprocess_ml("adata_1")         → adata_ml_2   (scale, HVG, PCA)
train_ml_models("adata_ml_2", models=["logistic_regression", "random_forest"])
visualize_tsne("adata_ml_2")     → PNGs colored by true / predicted / correct
```
Models: `logistic_regression`, `svm`, `random_forest`, `gradient_boosting`, `knn`,
`naive_bayes`, `neural_network` (PyTorch MLP).

**NLP annotation**
```
load_and_normalize("Spleen")     → adata_1
build_gene_rank_tokens("adata_1", top_k=100)
train_bilstm("adata_1", epochs=5)
run_order_experiment("adata_1")  → ordered vs shuffled + bar-chart PNG
```

**Compare both**
```
load_and_normalize(["Kidney", "Lung"])  → adata_1
compare_approaches("adata_1")            → ML best vs BiLSTM, side by side
```

### Example requests

- "Annotate the Heart with logistic regression and random forest, then show me the t-SNE."
- "Use the NLP approach on Spleen and test whether gene order matters."
- "Load Kidney and Lung and compare the ML and NLP approaches."

## Running

The agent and the scientific stack must share one environment. Use the `sc-a-env` conda env,
which has both (langgraph/langchain/groq + scanpy/anndata/scikit-learn/torch):

```bash
# put your Groq key in .env:  GROQ_API_KEY=...
/Users/naveenkumanan/miniconda3/envs/sc-a-env/bin/python build_compile.py
```

Edit the `HumanMessage` in `build_compile.py` (under `if __name__ == "__main__":`) to change
the request, or import `agent` from another script and call `agent.invoke(...)`.

## Data & outputs

- **Data:** `data/droplet/` (10X MTX) + `data/annotations/` (cell-type CSVs), one shared copy, not in git (run `scripts/get_data.sh`).
  Override with `SCRNA_DATA_ROOT` / `SCRNA_ANNOTATIONS_DIR`.
- **Figures:** written to `.scrna_figures/` (t-SNE plots, order-experiment chart).
- **Artifacts:** optional `.h5ad` snapshots under `.scrna_workspace/`.

## Notes

- Labels are joined to cells **by barcode** (`{channel}_{barcode}`), which is more robust than
  the notebooks' original positional/obs-name matching.
- Use `max_cells` on `load_and_normalize` to subsample for fast runs on large multi-tissue loads.
