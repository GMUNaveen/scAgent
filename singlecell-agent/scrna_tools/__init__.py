"""scrna_tools: callable pipeline pieces extracted from the two scRNA annotation projects.

- MyProject  -> Machine-Learning annotation (loading + ml_annotate + viz)
- NLPFinalProjectCode -> NLP / BiLSTM annotation (nlp_annotate), plus compare

Everything is designed so an LLM agent can call small, composable functions and pass
lightweight string *handles* between them (see workspace.py) instead of moving heavy
AnnData objects through the chat channel.
"""

from . import config, workspace, loading, ml_annotate, nlp_annotate, viz, compare

__all__ = [
    "config", "workspace", "loading", "ml_annotate", "nlp_annotate", "viz", "compare",
]
