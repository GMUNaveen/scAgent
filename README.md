# scAgent

A conversational agent for **scRNA-seq cell-type annotation**. Ask in plain English and the agent
picks a pipeline, runs it, and reports results. It offers two approaches on the same
[Tabula Muris](https://tabula-muris.ds.czbiohub.org/) 10X droplet data:

| Approach | What it does |
|---|---|
| **ML menu** | Per-tissue supervised classifiers (logistic regression, SVM, random forest, gradient boosting, kNN, naive Bayes, PyTorch MLP) on scaled / HVG / PCA features, with t-SNE plots of true vs. predicted labels |
| **NLP BiLSTM** | Each cell becomes a "sentence" of its top-K expressed genes, classified by a BiLSTM, plus an order-sensitivity experiment (ordered vs. shuffled genes) |

You can also ask it to **compare both** approaches head-to-head on the same cells.

## Repository layout

```
scAgent/
└── singlecell-agent/
    ├── build_compile.py    LangGraph agent (START → llm_call ⇄ tool_node → END)
    ├── model_node.py       LLM node (Groq, system prompt)
    ├── tool.py             @tool wrappers + model.bind_tools
    ├── tool_node.py        runs tool calls; errors are fed back to the LLM
    ├── end_logic.py        loop until the LLM stops calling tools
    ├── state.py            message state
    ├── scrna_tools/        the pipeline code (loading, ml_annotate, nlp_annotate, viz, compare)
    ├── MyProject/          original ML notebooks + data (used by the agent by default)
    └── NLPFinalProjectCode/ original NLP notebook, report and data
```

See [singlecell-agent/README.md](singlecell-agent/README.md) for architecture and tool-chain details.

## Quick start

1. **Environment** (needs langgraph/langchain/langchain-groq plus scanpy, anndata, scikit-learn, torch):
   ```bash
   conda activate sc-a-env   # or build your own from NLPFinalProjectCode/requirements.txt
   ```
2. **Data** — `cd singlecell-agent && scripts/get_data.sh` downloads the ~900 MB dataset from this repo's `data-v1` GitHub Release into `data/`.
3. **API key** — copy `singlecell-agent/.env.example` to `singlecell-agent/.env` and set `GROQ_API_KEY`.
4. **Run**:
   ```bash
   cd singlecell-agent
   python build_compile.py
   ```
   Edit the `HumanMessage` under `if __name__ == "__main__":` to change the request, or
   `from build_compile import agent` and call `agent.invoke({"messages": [...]})`.

### Example requests

- "Annotate the Heart with logistic regression and random forest, then show me the t-SNE."
- "Use the NLP approach on Spleen and test whether gene order matters."
- "Load Kidney and Lung and compare the ML and NLP approaches."

## Tools available to the agent

`list_available_tissues`, `load_and_normalize`, `preprocess_ml`, `train_ml_models`,
`visualize_tsne`, `build_gene_rank_tokens`, `train_bilstm`, `run_order_experiment`,
`compare_approaches` (plus demo arithmetic tools). Tools pass AnnData between each other by
short string **handles** (e.g. `adata_1`) so large matrices never go through the chat.

## Data and outputs

- Tissues: Bladder, Heart, Kidney, Liver, Lung, Mammary, Marrow, Muscle, Spleen, Thymus.
- Data: one shared copy in `singlecell-agent/data/` (`droplet/` + `annotations/`), **not stored in git**. `MyProject/` and `NLPFinalProjectCode/` symlink to it. Override with `SCRNA_DATA_ROOT`, `SCRNA_ANNOTATIONS_DIR`.
- Figures go to `singlecell-agent/.scrna_figures/`.

## Status

Verified working: all nine annotation tools run end to end, and the agent completes a full
Heart ML request via Groq (`openai/gpt-oss-120b`). On a 624-cell Heart run, logistic regression
reached ~99% test accuracy and the BiLSTM (2 epochs) ~53%, so more epochs are needed for the NLP path.

Known limitations:
- State lives in memory, so handles are lost when the process exits.
- The conversation is single-turn by default (no chat loop or memory across runs).
- The data is hosted as a release asset rather than in git; maintainers repack it with `scripts/pack_data.sh`.
- No automated tests yet.
