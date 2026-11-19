"""RDKit encoding of the graph inputs used by the published GraphEGFR weights."""

from rdkit import Chem


ELEMENTS = tuple(
    "C N O S F Si P Cl Br Mg Na Ca Fe As Al I B V K Tl Yb Sb Sn Ag Pd Co Se Ti "
    "Zn H Li Ge Cu Au Ni Cd In Mn Zr Cr Pt Hg Pb Unknown".split()
)
HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_STEREO = ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "unknown")


def one_hot(value, categories, use_last=False):
    if use_last and value not in categories:
        value = categories[-1]
    return [int(value == category) for category in categories]


def atom_features(atom):
    degree = atom.GetDegree()
    if degree > 10:
        raise ValueError(f"GraphEGFR does not support atom degree {degree}")
    return (
        one_hot(atom.GetSymbol(), ELEMENTS, use_last=True)
        + one_hot(degree, range(11))
        + one_hot(atom.GetValence(Chem.ValenceType.IMPLICIT), range(7), use_last=True)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS, use_last=True)
        + [int(atom.GetIsAromatic())]
        + one_hot(atom.GetTotalNumHs(), range(5), use_last=True)
    )


def molecule_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RenumberAtoms(mol, Chem.CanonicalRankAtoms(mol))
    # ConvMol sorts features by degree. The original bond graph keeps its
    # canonical indices, so keep these two orders for checkpoint compatibility.
    nodes = [atom_features(atom) for atom in sorted(mol.GetAtoms(), key=lambda atom: atom.GetDegree())]
    edges = []
    features = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((start, end), (end, start)))
        feature = (
            one_hot(str(bond.GetBondType()), BOND_TYPES)
            + [int(bond.IsInRing()), int(bond.GetIsConjugated())]
            + one_hot(str(bond.GetStereo()), BOND_STEREO, use_last=True)
        )
        features.extend((feature, feature))
    return nodes, edges, features
