"""ML annotation path (from MyProject notebooks).

Two stages, each an agent-callable tool:
  1. preprocess_ml : scale -> highly-variable genes -> subset -> PCA
  2. train_ml_models: label-encode -> train/test split -> fit the requested model menu

Faithful to the Heart/Liver/... notebooks: scaling happens before HVG selection
(flavor='seurat', min_disp=0.5), and the sklearn models use the same constructors.
An optional PyTorch MLP mirrors the notebook's 3-layer Classifier.
"""
from __future__ import annotations

import math

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from . import workspace


class _MLP(nn.Module):
    """The 3-layer fully-connected net from MyProject's PyTorch notebook cell."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TorchMLPClassifier:
    """sklearn-style wrapper (fit/predict) around the notebook's MLP so it fits MODEL_MENU."""

    def __init__(self, hidden: int = 10, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3):
        self.hidden, self.epochs, self.batch_size, self.lr = hidden, epochs, batch_size, lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(0)
        X = np.asarray(X, dtype=np.float32)
        num_classes = int(np.max(y)) + 1
        self.model = _MLP(X.shape[1], num_classes, self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(np.asarray(y), dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True,
        )
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self.model(xb.to(self.device)), yb.to(self.device))
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            return self.model(X).argmax(1).cpu().numpy()


# Canonical model menu — keys are what the agent selects from.
MODEL_MENU = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "svm": lambda: SVC(probability=True),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200),
    "gradient_boosting": lambda: GradientBoostingClassifier(),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": lambda: GaussianNB(),
    "neural_network": lambda: TorchMLPClassifier(),
}


def preprocess_ml(handle: str, min_disp: float = 0.5, n_comps: int = 50) -> dict:
    """Scale -> HVG -> subset to HVGs -> PCA. Creates a new artifact, returns its handle."""
    adata = workspace.get(handle).copy()

    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", max_mean=math.inf, min_disp=min_disp)
    adata = adata[:, adata.var["highly_variable"]].copy()
    # n_comps cannot exceed min(cells, genes)-1
    n_comps = int(min(n_comps, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.pca(adata, n_comps=n_comps)

    meta = {
        "stage": "ml_preprocessed",
        "from": handle,
        "n_cells": int(adata.n_obs),
        "n_hvg": int(adata.n_vars),
        "n_pcs": n_comps,
    }
    new_handle = workspace.put(adata, meta=meta, prefix="adata_ml")
    return {"handle": new_handle, **meta}


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def train_ml_models(
    handle: str,
    models: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    store_predictions: bool = True,
) -> dict:
    """Train the requested sklearn models on the HVG matrix and report test metrics.

    `models` is a subset of MODEL_MENU keys; None runs the full menu. When
    `store_predictions` is True, the best model is refit on all labeled cells and used to
    predict every cell; results are written to obs['ml_pred_label'] and obs['ml_correct']
    so downstream visualization can color a t-SNE by prediction / correctness.
    """
    adata = workspace.get(handle)
    if "cell_ontology_class" not in adata.obs or adata.obs["cell_ontology_class"].isna().all():
        raise ValueError(
            f"Artifact '{handle}' has no cell_ontology_class labels. "
            f"Load with attach_labels=True before training."
        )

    # Restrict to labeled cells.
    labeled = adata[adata.obs["cell_ontology_class"].notna()]
    X = _to_dense(labeled.X)
    y_raw = labeled.obs["cell_ontology_class"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if np.min(np.bincount(y)) >= 2 else None,
    )

    chosen = models or list(MODEL_MENU)
    unknown = [m for m in chosen if m not in MODEL_MENU]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Choose from {list(MODEL_MENU)}.")

    results = {}
    for name in chosen:
        clf = MODEL_MENU[name]()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "report": classification_report(
                y_test, preds, target_names=le.classes_, zero_division=0
            ),
        }

    ranked = sorted(results.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    best = ranked[0][0]

    if store_predictions:
        # Refit the best model on ALL labeled cells, then predict every cell in the artifact.
        clf = MODEL_MENU[best]()
        clf.fit(X, y)
        pred_all = clf.predict(_to_dense(adata.X))
        adata.obs["ml_pred_label"] = le.inverse_transform(pred_all)
        true = adata.obs["cell_ontology_class"].astype("object")
        correct = adata.obs["ml_pred_label"].astype("object") == true
        adata.obs["ml_correct"] = np.where(true.notna(), correct.astype(str), "unlabeled")
        workspace.update_meta(handle, prediction_model=best)

    return {
        "handle": handle,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "results": results,
        "best_model": best,
        "best_accuracy": ranked[0][1]["accuracy"],
        "prediction_model": best if store_predictions else None,
    }
