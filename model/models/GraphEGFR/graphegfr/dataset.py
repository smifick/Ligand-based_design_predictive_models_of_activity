import torch
import deepchem as dc
import json
from torch_geometric.data import Data, InMemoryDataset
import shutil
import os
from deepchem.data import NumpyDataset, DiskDataset
from typing import Optional
import numpy as np

class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass

class GraphData:

  def __init__(self,
               node_features: np.ndarray,
               edge_index: np.ndarray,
               edge_features: Optional[np.ndarray] = None,
               node_pos_features: Optional[np.ndarray] = None,
               **kwargs):

    # validate params
    if isinstance(node_features, np.ndarray) is False:
      raise ValueError('node_features must be np.ndarray.')

    if isinstance(edge_index, np.ndarray) is False:
      raise ValueError('edge_index must be np.ndarray.')
    elif issubclass(edge_index.dtype.type, np.integer) is False:
      raise ValueError('edge_index.dtype must contains integers.')
    elif edge_index.shape[0] != 2:
      raise ValueError('The shape of edge_index is [2, num_edges].')

    # np.max() method works only for a non-empty array, so size of the array should be non-zero
    elif (edge_index.size != 0) and (np.max(edge_index) >= len(node_features)):
      raise ValueError('edge_index contains the invalid node number.')

    if edge_features is not None:
      if isinstance(edge_features, np.ndarray) is False:
        raise ValueError('edge_features must be np.ndarray or None.')
      elif edge_index.shape[1] != edge_features.shape[0]:
        raise ValueError('The first dimension of edge_features must be the \
                          same as the second dimension of edge_index.')

    if node_pos_features is not None:
      if isinstance(node_pos_features, np.ndarray) is False:
        raise ValueError('node_pos_features must be np.ndarray or None.')
      elif node_pos_features.shape[0] != node_features.shape[0]:
        raise ValueError(
            'The length of node_pos_features must be the same as the \
                          length of node_features.')

    self.node_features = node_features
    self.edge_index = edge_index
    self.edge_features = edge_features
    self.node_pos_features = node_pos_features
    self.kwargs = kwargs
    self.num_nodes, self.num_node_features = self.node_features.shape
    self.num_edges = edge_index.shape[1]
    if self.edge_features is not None:
      self.num_edge_features = self.edge_features.shape[1]

    for key, value in self.kwargs.items():
      setattr(self, key, value)

  def __repr__(self) -> str:
    """Returns a string containing the printable representation of the object"""
    cls = self.__class__.__name__
    node_features_str = str(list(self.node_features.shape))
    edge_index_str = str(list(self.edge_index.shape))
    if self.edge_features is not None:
      edge_features_str = str(list(self.edge_features.shape))
    else:
      edge_features_str = "None"

    out = "%s(node_features=%s, edge_index=%s, edge_features=%s" % (
        cls, node_features_str, edge_index_str, edge_features_str)
    # Adding shapes of kwargs
    for key, value in self.kwargs.items():
      out += (', ' + key + '=' + str(list(value.shape)))
    out += ')'
    return out

  def to_dgl_graph(self, self_loop: bool = False):

    try:
      import dgl
      import torch
    except ModuleNotFoundError:
      raise ImportError("This function requires DGL to be installed.")

    src = self.edge_index[0]
    dst = self.edge_index[1]

    g = dgl.graph((torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
                  num_nodes=self.num_nodes)
    g.ndata['x'] = torch.from_numpy(self.node_features).float()

    if self.node_pos_features is not None:
      g.ndata['pos'] = torch.from_numpy(self.node_pos_features).float()

    if self.edge_features is not None:
      g.edata['edge_attr'] = torch.from_numpy(self.edge_features).float()

    if self_loop:
      # This assumes that the edge features for self loops are full-zero tensors
      # In the future we may want to support featurization for self loops
      g.add_edges(np.arange(self.num_nodes), np.arange(self.num_nodes))

    return g


def load_dataset(adj,smiles,Y,fpf,fpc):
    smiles_list = []
    d = []
    graphconvfeaturizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    conv_mol_featurizer = dc.feat.ConvMolFeaturizer()
    arr_convmol = conv_mol_featurizer.featurize(smiles)
    arr_graph = graphconvfeaturizer.featurize(smiles)
    for i in range(len(Y)):
        x = torch.Tensor(arr_convmol[i].get_atom_features())
        edge_index = torch.LongTensor(arr_graph[i].edge_index)
        edge_attr = torch.LongTensor(arr_graph[i].edge_features)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=Y[i], adj = adj[i], fpf=fpf[i], fpc=fpc[i], \
                    dgl=GraphData(arr_convmol[i].get_atom_features(), \
                    arr_graph[i].edge_index, arr_graph[i].edge_features).to_dgl_graph())
        d.append(data)
        smiles_list.append(smiles[i])
    set_ = TestDataset(d)
    return set_, smiles_list

class KFoldDataset(object):
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def create_dataset(self, dc_tvset,folds,splitter,seed):
        os.makedirs(f'{self.run_dir}/.tmpfolder/',exist_ok=True)
        directories = []
        for fold in range(folds):
            directories.extend([f'{self.run_dir}/.tmpfolder/{fold}-train',f'{self.run_dir}/.tmpfolder/{fold}-test'])
        for fold, (dc_trainset, dc_validset) in enumerate(splitter.k_fold_split(dc_tvset,k=folds,seed=seed,directories=directories)):
            pass

    def get_dataset(self, fold):
        trainset = NumpyDataset.from_DiskDataset(DiskDataset(data_dir=f'{self.run_dir}/.tmpfolder/{fold}-train'))
        testset = NumpyDataset.from_DiskDataset(DiskDataset(data_dir=f'{self.run_dir}/.tmpfolder/{fold}-test'))

        return trainset, testset

    def delete(self):
        shutil.rmtree(f'{self.run_dir}/.tmpfolder/')
