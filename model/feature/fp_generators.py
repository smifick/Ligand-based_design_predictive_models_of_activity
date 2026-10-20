
from __future__ import annotations

from typing import Callable

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

FpCallable = Callable[[Chem.Mol], object]
FpFactory = Callable[[int, int, bool], FpCallable]


def make_morgan(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    kwargs = {"radius": radius, "fpSize": n_bits}
    if use_features:
        kwargs["atomInvariantsGenerator"] = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
    gen = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
    return gen.GetFingerprint


def make_rdkit(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits).GetFingerprint


def make_atompair(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits).GetFingerprint


def make_torsion(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=n_bits
    ).GetFingerprint


def make_maccs(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    # MACCS всегда 167 бит — n_bits игнорируется
    return MACCSkeys.GenMACCSKeys


def make_avalon(radius: int, n_bits: int, use_features: bool) -> FpCallable:
    from rdkit.Avalon import pyAvalonTools

    return lambda mol: pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
