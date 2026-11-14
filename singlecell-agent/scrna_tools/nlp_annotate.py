"""NLP annotation path (from NLPFinalProjectCode/FinalProjectNotebook.ipynb).

Frames each cell as a "sentence" of its top-K highest-expressed gene symbols (ranked by
expression), then classifies cell type with a BiLSTM. Includes the order-sensitivity
control experiment (train on ordered vs. shuffled gene sequences).

Three agent-callable stages, all branching off the SHARED load_and_normalize handle:
  1. build_gene_rank_tokens : top-K gene sentences -> vocab -> encoded X_ids (+ labels)
  2. train_bilstm           : train/eval the BiLSTM classifier
  3. run_order_experiment   : ordered vs shuffled comparison (+ bar-chart PNG)

The notebook re-derived labels via a bespoke obs_name match; here we reuse the labels the
shared loader already attached (cell_ontology_class), so both paths share one source of truth.
"""
from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from . import config, workspace  # noqa: E402

SPECIAL_TOKENS = ["<pad>", "<unk>"]
# BiLSTM hyperparameters (match the notebook)
EMB_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 256
EPOCHS_DEFAULT = 5
LR = 1e-3


# ─────────────────────────────── tokenization ───────────────────────────────
def topk_gene_indices(row, k: int) -> np.ndarray:
    """Indices of the top-k highest-expressed genes in a cell row (sparse or dense)."""
    if sp.issparse(row):
        row = row.tocsr()
        data, idx = row.data, row.indices
        if data.size == 0:
            return np.array([], dtype=int)
        if data.size > k:
            top = np.argpartition(data, -k)[-k:]
            order = top[np.argsort(data[top])[::-1]]
            return idx[order]
        order = np.argsort(data)[::-1]
        return idx[order]
    arr = np.asarray(row).ravel()
    if arr.size == 0:
        return np.array([], dtype=int)
    if k < arr.size:
        idx = np.argpartition(arr, -k)[-k:]
        idx = idx[np.argsort(arr[idx])[::-1]]
    else:
        idx = np.argsort(arr)[::-1]
    return idx[arr[idx] > 0]


def build_gene_rank_tokens(handle: str, top_k: int = 100, save_corpus: bool = False) -> dict:
    """Turn each labeled cell into a top-K gene sentence, build vocab, and encode.

    Stores the NLP artifacts (X_ids, y, vocab, label_names, top_k) in adata.uns['nlp'].
    """
    adata = workspace.get(handle)
    if "cell_ontology_class" not in adata.obs or adata.obs["cell_ontology_class"].isna().all():
        raise ValueError(
            f"Artifact '{handle}' has no labels. Load with attach_labels=True first."
        )

    labeled_mask = adata.obs["cell_ontology_class"].notna().values
    sub = adata[labeled_mask]
    X = sub.X
    var_names = np.array(sub.var_names)

    sentences = []
    for i in range(sub.n_obs):
        idx = topk_gene_indices(X[i], top_k)
        sentences.append(" ".join(var_names[idx]))

    labels = sub.obs["cell_ontology_class"].astype(str).tolist()

    # Vocab (pad=0, unk=1, then sorted gene symbols)
    all_tokens = {tok for s in sentences for tok in s.split()}
    vocab = {tok: i + len(SPECIAL_TOKENS) for i, tok in enumerate(sorted(all_tokens))}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i

    def encode(s: str) -> list[int]:
        ids = [vocab.get(t, vocab["<unk>"]) for t in s.split()[:top_k]]
        return ids + [vocab["<pad>"]] * (top_k - len(ids))

    X_ids = np.array([encode(s) for s in sentences], dtype=np.int64)

    label_set = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(label_set)}
    y = np.array([label_to_id[lab] for lab in labels], dtype=np.int64)

    adata.uns["nlp"] = {
        "X_ids": X_ids,
        "y": y,
        "vocab": vocab,
        "label_names": label_set,
        "top_k": top_k,
        "pad_idx": vocab["<pad>"],
    }
    workspace.update_meta(handle, nlp_tokens=True, nlp_top_k=top_k)

    if save_corpus:
        config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        out = config.WORKSPACE_DIR / f"gene_rank_sentences_{handle}.txt"
        out.write_text("\n".join(sentences) + "\n")

    return {
        "handle": handle,
        "n_sentences": int(len(sentences)),
        "vocab_size": int(len(vocab)),
        "num_classes": int(len(label_set)),
        "top_common_labels": Counter(labels).most_common(5),
    }


# ─────────────────────────────── model ───────────────────────────────
class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.emb(x)
        _, (h_n, _) = self.lstm(emb)
        pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)  # order-sensitive readout
        return self.fc(pooled)


def _split(X_ids, y):
    """70/15/15 stratified split, falling back to unstratified if a class is too small."""
    strat1 = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_ids, y, test_size=0.3, random_state=0, stratify=strat1
    )
    strat2 = y_tmp if (strat1 is not None and np.min(np.bincount(y_tmp)) >= 2) else None
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=strat2
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def _run_eval(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, 1).cpu().numpy())
            trues.append(yb.numpy())
    yt, yp = np.concatenate(trues), np.concatenate(preds)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0)


def _train_once(X_ids, y, vocab_size, num_classes, pad_idx, epochs):
    """Train a fresh BiLSTM and return (test_acc, test_f1, per_epoch_val)."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X_ids, y)

    train_loader = DataLoader(GeneDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GeneDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(GeneDataset(X_te, y_te), batch_size=BATCH_SIZE)

    model = BiLSTMClassifier(vocab_size, EMB_DIM, HIDDEN_DIM, num_classes, pad_idx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    per_epoch = []
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        per_epoch.append(_run_eval(model, val_loader, device))
    acc, f1 = _run_eval(model, test_loader, device)
    return acc, f1, per_epoch


def _require_tokens(handle: str) -> dict:
    adata = workspace.get(handle)
    if "nlp" not in adata.uns:
        raise ValueError(
            f"Artifact '{handle}' has no gene-rank tokens. Run build_gene_rank_tokens first."
        )
    return adata.uns["nlp"]


def train_bilstm(handle: str, epochs: int = EPOCHS_DEFAULT) -> dict:
    """Train the BiLSTM on the gene-rank tokens and report test accuracy / macro-F1."""
    nlp = _require_tokens(handle)
    acc, f1, per_epoch = _train_once(
        nlp["X_ids"], nlp["y"], len(nlp["vocab"]), len(nlp["label_names"]),
        nlp["pad_idx"], epochs,
    )
    return {
        "handle": handle,
        "epochs": epochs,
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1),
        "val_by_epoch": [{"acc": float(a), "f1": float(f)} for a, f in per_epoch],
        "num_classes": int(len(nlp["label_names"])),
    }


def shuffle_sequences(X_ids, pad_idx=0, seed=0):
    """Randomize the order of non-pad tokens within each row (keeps the gene *set*)."""
    rng = np.random.default_rng(seed)
    X_shuf = X_ids.copy()
    for i in range(X_shuf.shape[0]):
        row = X_shuf[i]
        mask = row != pad_idx
        tokens = row[mask]
        rng.shuffle(tokens)
        row[mask] = tokens
    return X_shuf


def run_order_experiment(handle: str, epochs: int = EPOCHS_DEFAULT) -> dict:
    """Train the BiLSTM on ordered vs. shuffled gene sequences to test if order matters."""
    nlp = _require_tokens(handle)
    vocab_size, num_classes, pad_idx = len(nlp["vocab"]), len(nlp["label_names"]), nlp["pad_idx"]

    acc_o, f1_o, _ = _train_once(nlp["X_ids"], nlp["y"], vocab_size, num_classes, pad_idx, epochs)
    X_shuf = shuffle_sequences(nlp["X_ids"], pad_idx=pad_idx, seed=0)
    acc_s, f1_s, _ = _train_once(X_shuf, nlp["y"], vocab_size, num_classes, pad_idx, epochs)

    # Bar-chart comparison (mirrors the notebook's final plot)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.175, [acc_o, acc_s], 0.35, label="Accuracy")
    ax.bar(x + 0.175, [f1_o, f1_s], 0.35, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ordered", "Shuffled"])
    ax.set_ylim(0, 1)
    ax.set_title("Order Sensitivity: Ordered vs Shuffled")
    ax.legend()
    fig_path = config.FIGURES_DIR / f"order_experiment_{handle}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "handle": handle,
        "epochs": epochs,
        "ordered": {"accuracy": float(acc_o), "macro_f1": float(f1_o)},
        "shuffled": {"accuracy": float(acc_s), "macro_f1": float(f1_s)},
        "accuracy_drop": float(acc_o - acc_s),
        "figure": str(fig_path),
    }
