from __future__ import annotations

from typing import Callable

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

FpCallable = Callable[[Chem.Mol], object]
FpFactory = Callable[[dict], FpCallable]

MORGAN_DEFAULTS = {
    "radius": 2,
    "n_bits": 2048,
    "use_features": False,
    "include_chirality": False,
    "use_bond_types": True,
}

RDKIT_DEFAULTS = {
    "min_path": 1,
    "max_path": 7,
    "n_bits": 2048,
    "use_hs": True,
    "branched_paths": True,
    "use_bond_order": True,
    "num_bits_per_feature": 2,
}

ATOMPAIR_DEFAULTS = {
    "min_distance": 1,
    "max_distance": 30,
    "n_bits": 2048,
    "include_chirality": False,
    "use_2d": True,
}

TORSION_DEFAULTS = {
    "torsion_atom_count": 4,
    "n_bits": 2048,
    "include_chirality": False,
}

MACCS_DEFAULTS: dict = {}

AVALON_DEFAULTS = {
    "n_bits": 512,
    "is_query": False,
    "bit_flags": 15761407,
}

def _apply(defaults: dict, params: dict) -> dict:
    unknown = set(params) - set(defaults)
    if unknown:
        known = ", ".join(sorted(defaults)) or "нет"
        raise ValueError(
            f"Неизвестные параметры: {', '.join(sorted(unknown))}. Допустимы: {known}"
        )
    resolved = dict(defaults)
    resolved.update(params)
    return resolved


def make_morgan(params: dict) -> FpCallable:
    resolved = _apply(MORGAN_DEFAULTS, params)
    kwargs = {
        "radius": resolved["radius"],
        "fpSize": resolved["n_bits"],
        "includeChirality": resolved["include_chirality"],
        "useBondTypes": resolved["use_bond_types"],
    }
    if resolved["use_features"]:
        kwargs["atomInvariantsGenerator"] = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
    return rdFingerprintGenerator.GetMorganGenerator(**kwargs).GetFingerprint


def make_rdkit(params: dict) -> FpCallable:
    resolved = _apply(RDKIT_DEFAULTS, params)
    return rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=resolved["min_path"],
        maxPath=resolved["max_path"],
        useHs=resolved["use_hs"],
        branchedPaths=resolved["branched_paths"],
        useBondOrder=resolved["use_bond_order"],
        fpSize=resolved["n_bits"],
        numBitsPerFeature=resolved["num_bits_per_feature"],
    ).GetFingerprint


def make_atompair(params: dict) -> FpCallable:
    resolved = _apply(ATOMPAIR_DEFAULTS, params)
    return rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=resolved["min_distance"],
        maxDistance=resolved["max_distance"],
        includeChirality=resolved["include_chirality"],
        use2D=resolved["use_2d"],
        fpSize=resolved["n_bits"],
    ).GetFingerprint


def make_torsion(params: dict) -> FpCallable:
    resolved = _apply(TORSION_DEFAULTS, params)
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=resolved["include_chirality"],
        torsionAtomCount=resolved["torsion_atom_count"],
        fpSize=resolved["n_bits"],
    ).GetFingerprint


def make_maccs(params: dict) -> FpCallable:
    _apply(MACCS_DEFAULTS, params)
    return MACCSkeys.GenMACCSKeys


def make_avalon(params: dict) -> FpCallable:
    resolved = _apply(AVALON_DEFAULTS, params)
    from rdkit.Avalon import pyAvalonTools

    n_bits = resolved["n_bits"]
    is_query = resolved["is_query"]
    bit_flags = resolved["bit_flags"]

    def _fp(mol: Chem.Mol):
        return pyAvalonTools.GetAvalonFP(
            mol,
            nBits=n_bits,
            isQuery=is_query,
            bitFlags=bit_flags,
        )

    return _fp


# A None factory marks the descriptor that passes the whole input through.
_DESCRIPTOR_SPECS: dict[str, tuple[dict, FpFactory | None]] = {
    "identity": ({}, None),
    "morgan": (MORGAN_DEFAULTS, make_morgan),
    "rdkit": (RDKIT_DEFAULTS, make_rdkit),
    "atompair": (ATOMPAIR_DEFAULTS, make_atompair),
    "torsion": (TORSION_DEFAULTS, make_torsion),
    "maccs": (MACCS_DEFAULTS, make_maccs),
    "avalon": (AVALON_DEFAULTS, make_avalon),
}

FP_GENERATORS: dict[str, FpFactory] = {
    name: factory
    for name, (_, factory) in _DESCRIPTOR_SPECS.items()
    if factory is not None
}

DESCRIPTOR_DEFAULTS = {
    name: defaults for name, (defaults, _) in _DESCRIPTOR_SPECS.items()
}

COMMON_DESCRIPTOR_DEFAULTS = {
    "smiles_col": "std_smiles",
    "n_jobs": 4,
    "on_error": "raise",
    "verbose": 0,
}

DESCRIPTOR_REGISTRY = {
    name: {"name": name, "default": dict(defaults)}
    for name, defaults in DESCRIPTOR_DEFAULTS.items()
}
