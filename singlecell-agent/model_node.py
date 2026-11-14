from langchain.messages import SystemMessage
from tool import model_with_tools


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a single-cell RNA annotation assistant. Help the user "
                            "annotate scRNA-seq data using the provided tools. Always start with "
                            "load_and_normalize(tissue) to get a handle; every path builds on it. "
                            "Pass the handle returned by one step into the next.\n"
                            "Two annotation approaches are available (the user may pick either):\n"
                            "- ML path: preprocess_ml(handle) -> train_ml_models(handle) -> "
                            "optionally visualize_tsne(handle) for t-SNE plots of true vs "
                            "predicted cell types.\n"
                            "- NLP path: build_gene_rank_tokens(handle) -> train_bilstm(handle), "
                            "and optionally run_order_experiment(handle) to test whether gene "
                            "order carries signal.\n"
                            "To compare both approaches on the same data, call "
                            "compare_approaches(handle) after load_and_normalize.\n"
                            "Use list_available_tissues if the user hasn't picked a tissue. "
                            "Arithmetic tools are also available."
                        )
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }