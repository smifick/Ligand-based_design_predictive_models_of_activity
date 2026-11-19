"""Joblib-serializable GraphEGFR predictor for the shared inference runner."""

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np


FINGERPRINT_TYPES = {
    "daylightFP": "daylight", "extendedFP": "extended", "graphFP": "graph",
    "PubchemFP": "pubchem", "estateFP": "estate", "hybridizationFP": "hybridization",
    "lingoFP": "lingo", "klekota_rothFP": "klekota-roth", "circularFP": "circular",
    "rdkitFP": "rdkit", "maccsFP": "maccs", "AvalonFP": "Avalon",
    "AtomPairFP": "AtomPair", "FP2FP": "FP2", "FP3FP": "FP3", "FP4FP": "FP4",
}
NON_CDK_TYPES = {"rdkit", "maccs", "Avalon", "FP2", "FP3", "FP4"}
DEFAULT_CDK_JAR = Path(__file__).resolve().parents[3] / "data" / "weights" / "graphegfr" / "cdk-2.3.jar"


def fingerprint_layout(columns):
    layout = []
    for column in columns:
        match = re.fullmatch(r"(.+FP)([0-9]+)", column)
        if match is None or match[1] not in FINGERPRINT_TYPES:
            raise ValueError(f"Unknown GraphEGFR fingerprint column: {column!r}")
        layout.append((FINGERPRINT_TYPES[match[1]], int(match[2])))
    return tuple(layout)


def cdk_package():
    import jpype

    jar = Path(os.environ.get("GRAPHEGFR_CDK_JAR", str(DEFAULT_CDK_JAR))).resolve()
    if not jar.is_file():
        raise FileNotFoundError(f"CDK JAR not found at {jar}. Set GRAPHEGFR_CDK_JAR to cdk-2.3.jar.")
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[str(jar)])
    elif not hasattr(jpype.JPackage("org.openscience.cdk"), "DefaultChemObjectBuilder"):
        jpype.addClassPath(str(jar))
    return jpype.JPackage("org.openscience.cdk")


def fingerprint_matrix(smiles, layout, cdk):
    from .graphegfr.fingerprint import convert_bitvect_to_numpy_array, get_fingerprint

    rows = []
    for value in smiles:
        vectors = {}
        for kind, index in layout:
            if kind not in vectors:
                bits = get_fingerprint(cdk, value, fp_type=kind)
                vectors[kind] = convert_bitvect_to_numpy_array(bits, kind)
        rows.append([vectors[kind][index] for kind, index in layout])
    return np.asarray(rows, dtype=np.float32)


def graph_features(smiles):
    import torch

    from .graphegfr.graph_features import molecule_graph

    node_features, edge_indices, edge_features, membership = [], [], [], []
    offset = 0
    for index, value in enumerate(smiles):
        nodes, edges, bonds = molecule_graph(value)
        count = len(nodes)
        node_features.append(torch.tensor(nodes, dtype=torch.float32))
        edge_indices.append(torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T + offset)
        edge_features.append(torch.tensor(bonds, dtype=torch.float32).reshape(-1, 11))
        membership.append(torch.full((count,), index, dtype=torch.long))
        offset += count
    return {
        "x": torch.cat(node_features),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_features),
        "batch": torch.cat(membership),
        "num_graphs": len(smiles),
    }


class GraphEGFRPredictor:
    def __init__(self, adapter, hash_columns, nonhash_columns, output_index, batch_size=32):
        self.adapter = adapter
        self.hash_layout = fingerprint_layout(hash_columns)
        self.nonhash_layout = fingerprint_layout(nonhash_columns)
        self.output_index = output_index
        self.batch_size = batch_size

        model = adapter.model
        if len(self.hash_layout) != model.fingfeaf or len(self.nonhash_layout) != model.fingfeac:
            raise ValueError("Fingerprint CSV widths must match the checkpoint input dimensions.")
        if not 0 <= output_index < model.n_tasks:
            raise ValueError("Output index is outside the checkpoint output dimensions.")
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")

    def predict(self, X):
        import torch
        from rdkit import Chem
        from rdkit.Chem.SaltRemover import SaltRemover

        model = self.adapter.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.adapter.device = device
        active_layout = (self.hash_layout if model.enable_feaf else ()) + (
            self.nonhash_layout if model.enable_feac else ()
        )
        cdk = cdk_package() if any(kind not in NON_CDK_TYPES for kind, _ in active_layout) else None
        remover = SaltRemover()
        predictions = []
        source = X["std_smiles"]
        for start in range(0, len(source), self.batch_size):
            smiles = []
            for offset, value in enumerate(source.iloc[start:start + self.batch_size]):
                mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
                mol = remover.StripMol(mol) if mol is not None else None
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError(f"Invalid or empty SMILES at input row {start + offset}.")
                smiles.append(Chem.MolToSmiles(mol))
            batch = graph_features(smiles) if model.enable_fea else {}
            if model.enable_feaf:
                batch["fpf"] = torch.from_numpy(fingerprint_matrix(smiles, self.hash_layout, cdk))
            if model.enable_feac:
                batch["fpc"] = torch.from_numpy(fingerprint_matrix(smiles, self.nonhash_layout, cdk))
            batch = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
            result = self.adapter.predict(SimpleNamespace(**batch))
            predictions.append(result[:, self.output_index].cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty(0)
