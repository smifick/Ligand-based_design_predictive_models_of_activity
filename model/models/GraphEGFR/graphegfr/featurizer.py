# %%
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import deepchem as dc
import rdkit
from rdkit import Chem
import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
import pickle
import argparse
import os
import time
import numpy as np
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import MolFromSmiles,MolToSmiles
# %%
def get_csv(path, smipos):
    df = pd.read_csv(path)
    df= df.rename(columns={smipos:'SMILES_NS'})
    return df

def clean_smiles(smiles: pd.Series) -> pd.Series:
    if type(smiles) in [list, np.ndarray]:
        smiles = pd.Series(smiles)
    remover = SaltRemover()
    len(remover.salts)

    SMILES_desalt = []
    defect = []
    
    for i in smiles.tolist():
        mol = MolFromSmiles(i) 
        if mol != None:    
            mol_desalt = remover.StripMol(mol)
            mol_SMILES = MolToSmiles(mol_desalt)
            SMILES_desalt.append(mol_SMILES)
        else:
            defect.append(i)
    print("Number of defect:",len(defect))
    if len(defect) != 0:
        print("== List ==")
        for j in defect:
            print(j)
    return pd.Series(SMILES_desalt)

def preprocess_graph(data): #The function is to preprocessed the adjacency matrix, returning the normalized adjacency matrix in the form of numpy array for feeding into the model
    adj_ = data + sp.eye(data.shape[0]) 
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return np.array(adj_normalized)

def smiles_get_features(a): #This function will return the smiles code into list of feature for each atoms
    if isinstance(a, float):
        return pd.np.nan
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    atom_features = features.get_atom_features() # initial atom feature vectors
    if atom_features.shape[0] > 60:
        return pd.np.nan
    return atom_features

def smiles_get_adj(a): #This function retrieve the adjacency matrix from the molecule
    if isinstance(a, float):
        return pd.np.nan
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    adj_list = features.get_adjacency_list() # adjacency list (neighbor list)
    adj=np.zeros((len(adj_list), len(adj_list))) # convert adjacency list into adjacency matrix "A"
    if len(adj_list) > 60:
        return pd.np.nan
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            adj[i][j]=1
    return adj

def smiles_get_edge(a): #This function retrieve the adjacency matrix from the molecule
    if isinstance(a, float):
        return pd.np.nan
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    adj_list = features.get_adjacency_list() # adjacency list (neighbor list)
    node1 = []
    node2 = []
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            node1.append(i)
            node2.append(j)
    return np.stack((np.array(node1),np.array(node2)))

def sim_graph(smile):
    if isinstance(smile, float):
        return pd.np.nan
    mol = rdkit.Chem.MolFromSmiles(smile)
    if mol is None:
        return pd.np.nan
    Chem.Kekulize(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    am = Chem.GetAdjacencyMatrix(mol,useBO=True)
    if len(atoms)>60:
        return pd.np.nan
    for i,atom in enumerate(atoms):
        am[i,i] = atom
    return am

def get_max_dim(d): #This funcion is used to find the maximum dimension the set of data contain
    maxdim = 0
    for i in d:
        if i.shape[0]>maxdim:
            maxdim = i.shape[0]
    return maxdim

def pad_up_to(t, max_in_dims, constant_values=0): #This function is used to pad the data up to a given dimension
    s = t.shape
    size = np.subtract(max_in_dims, s)
    return np.pad(t, ((0,size[0]),(0,size[1])), 'constant', constant_values=constant_values)

def get_feature(smi: pd.Series, max_dim, path, save = True):
    pre_feature = smi.apply(smiles_get_features).dropna()
    True_feature= pre_feature.apply(pad_up_to, args=((max_dim, 75),))
    True_array_feature = np.stack(True_feature.values)
    if save:
        np.save(path+'/feature.npy',True_array_feature, fix_imports = False)
    return True_array_feature

def get_graph(smi: pd.Series, max_dim, path, save = True):
    pre_graph = smi.apply(sim_graph).dropna()
    true_graph = pre_graph.apply(pad_up_to, args=((max_dim, max_dim),))
    true_array_graph = np.stack(true_graph.values)
    if save:
        np.save(path+'/graph',true_array_graph, fix_imports = False)
    return true_array_graph

# %%
def generate_npdata(smi: pd.Series,  path, save = True):
    os.makedirs(path, exist_ok=True)
    pre_adj = smi.apply(smiles_get_adj)
    edge = smi.apply(smiles_get_edge).rename('edge').tolist()
    
    pre_adj = pre_adj.dropna()
    max_dim = 60
    True_adj= pre_adj.apply(preprocess_graph)
    adj = np.stack(True_adj.apply(pad_up_to, args=((max_dim, max_dim),)).values)
    if save:
        with open(path + '/edge.npy', 'wb') as f:
            pickle.dump(edge, f)
        np.save(path+'/adj.npy',adj, fix_imports = False)
        
    feature = get_feature(smi, max_dim, path, save)
    graph = get_graph(smi, max_dim, path, save)
    return [adj, feature, graph, edge]
    