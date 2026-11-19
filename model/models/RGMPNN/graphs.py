"""Build and batch the two graphs expected by upstream RGNN."""
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from .upstream.atom_feature import atom_feature
from .upstream.bond_feature import bond_feature
from .upstream.reduced_features import rg_feature, rg_x_feature


class AtomGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "pool_index":
            # The two rows index different node sets.
            return value.new_tensor([[self.num_nodes], [self.num_reduced_nodes]])
        return super().__inc__(key, value, *args, **kwargs)


def smiles_to_graphs(smiles, position):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Missing SMILES at row {position}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES at row {position}: {smiles!r}")

    x = atom_feature(mol, atom_types=118)
    edge_index, edge_attr = bond_feature(mol, bond_type_num=4)
    groups, membership, rg_edges = rg_feature(mol, edge_index)
    atoms = AtomGraph(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        pool_index=membership[:, [1, 0]].t().contiguous(),
        num_reduced_nodes=len(groups),
    )
    reduced = Data(
        x=rg_x_feature(groups),
        edge_index=rg_edges[:, :2].t().contiguous(),
        edge_attr=rg_edges[:, 2:].float(),
    )
    return atoms, reduced


def collate_graphs(graphs):
    atoms, reduced = zip(*graphs)
    return Batch.from_data_list(list(atoms)), Batch.from_data_list(list(reduced))

