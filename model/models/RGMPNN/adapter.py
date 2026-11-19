"""Scikit-learn adapter for ligand-based pIC50 regression."""
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class RGMPNNRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, *, epochs=50, batch_size=32, learning_rate=0.001,
        hidden_channels=128, dropout=0.1, random_state=67,
        device="cpu", smiles_col="std_smiles",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.random_state = random_state
        self.device = device
        self.smiles_col = smiles_col

    def _graphs(self, X):
        from .graphs import smiles_to_graphs

        if hasattr(X, "columns"):
            if self.smiles_col not in X.columns:
                raise ValueError(f"Missing SMILES column {self.smiles_col!r}")
            X = X[self.smiles_col]
        smiles = np.asarray(X, dtype=object)
        if smiles.ndim == 2 and smiles.shape[1] == 1:
            smiles = smiles[:, 0]
        if smiles.ndim != 1:
            raise ValueError("RG-MPNN expects a one-dimensional sequence of SMILES")
        return [smiles_to_graphs(value, i) for i, value in enumerate(smiles)]

    def fit(self, X, y, *, validation_data=None, on_validation=None):
        import torch
        from torch.utils.data import DataLoader

        from model.evalution.evalution import regression_metrics

        from .graphs import collate_graphs
        from .upstream.network import RGNN

        for name in ("epochs", "batch_size", "hidden_channels"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim != 1 or not targets.size or not np.isfinite(targets).all():
            raise ValueError("pIC50 targets must be a nonempty vector of finite values")
        if len(X) != targets.size:
            raise ValueError("SMILES and pIC50 targets must have the same length")
        graphs = self._graphs(X)
        mean = float(targets.mean())
        scale = float(targets.std()) or 1.0
        labels = torch.from_numpy((targets - mean) / scale)

        validation_graphs = None
        validation_history = []
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val = np.asarray(y_val, dtype=float)
            if y_val.shape != (len(X_val),) or not y_val.size or not np.isfinite(y_val).all():
                raise ValueError("Validation requires a nonempty vector of finite pIC50 targets matching the SMILES")
            validation_graphs = self._graphs(X_val)

        device = torch.device(self.device)
        rng = torch.Generator()
        if self.random_state is None:
            rng.seed()
        else:
            rng.manual_seed(int(self.random_state))
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

        # Keep initialization and dropout reproducible without changing caller RNG state.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(rng.initial_seed())
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(rng.initial_seed())
            atom_graph, _ = graphs[0]
            model = RGNN(
                in_channels=atom_graph.x.shape[1], channels=self.hidden_channels,
                out_channels=1, edge_dim=atom_graph.edge_attr.shape[1],
                num_passing_atom=3, num_passing_pool=1, num_passing_rg=1,
                num_passing_mol=1, dropout=self.dropout,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loader = DataLoader(
                range(len(graphs)), batch_size=self.batch_size, shuffle=True,
                generator=rng,
            )
            for epoch in range(1, self.epochs + 1):
                model.train()
                for indices in loader:
                    atoms, reduced = collate_graphs([graphs[i] for i in indices.tolist()])
                    optimizer.zero_grad()
                    prediction, _ = model(atoms.to(device), reduced.to(device))
                    loss = torch.nn.functional.mse_loss(prediction, labels[indices].to(device))
                    loss.backward()
                    optimizer.step()

                if validation_graphs is not None:
                    model.eval()
                    predictions = self._predict_graphs(model, validation_graphs, device)
                    metrics = regression_metrics(y_val, predictions * scale + mean)
                    validation_history.append(metrics)
                    if on_validation is not None:
                        on_validation(epoch, metrics.copy())

        # Store CPU weights in joblib artifacts, including after CUDA training.
        self.model_ = model.cpu().eval()
        self.target_mean_ = mean
        self.target_scale_ = scale
        self.validation_history_ = validation_history
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        predictions = self._predict_graphs(self.model_, self._graphs(X), "cpu")
        return predictions * self.target_scale_ + self.target_mean_

    def _predict_graphs(self, model, graphs, device):
        import torch

        from .graphs import collate_graphs

        if not graphs:
            return np.empty(0, dtype=np.float32)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(graphs), self.batch_size):
                atoms, reduced = collate_graphs(graphs[start:start + self.batch_size])
                prediction, _ = model(atoms.to(device), reduced.to(device))
                predictions.append(prediction.cpu().numpy())
        return np.concatenate(predictions)

