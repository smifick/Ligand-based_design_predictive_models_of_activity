"""Tensor implementation of AttentiveFP with DGL-LifeSci 0.3.2 parameter names."""

import torch
from torch import nn
from torch.nn import functional as F


def segment_sum(values, groups, count):
    return values.new_zeros((count, values.size(1))).index_add(0, groups, values)


def segment_softmax(logits, groups, count):
    maxima = logits.new_full((count, 1), -torch.inf)
    maxima = maxima.scatter_reduce(0, groups[:, None], logits, reduce="amax")
    weights = (logits - maxima[groups]).exp()
    return weights / segment_sum(weights, groups, count)[groups]


class InitialGRU(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.edge_transform = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, size))
        self.gru = nn.GRUCell(size, size)

    def forward(self, dst, logits, edges, nodes):
        weights = segment_softmax(logits, dst, len(nodes))
        context = segment_sum(weights * self.edge_transform(edges), dst, len(nodes))
        return F.relu(self.gru(F.elu(context), nodes))


class InitialContext(nn.Module):
    def __init__(self, node_size, edge_size, size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(nn.Linear(node_size, size), nn.LeakyReLU())
        self.project_edge1 = nn.Sequential(nn.Linear(node_size + edge_size, size), nn.LeakyReLU())
        self.project_edge2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * size, 1), nn.LeakyReLU())
        self.attentive_gru = InitialGRU(size, dropout)

    def forward(self, src, dst, nodes, edges):
        node_states = self.project_node(nodes)
        edge_states = self.project_edge1(torch.cat((nodes[src], edges), dim=1))
        logits = self.project_edge2(torch.cat((node_states[dst], edge_states), dim=1))
        return self.attentive_gru(dst, logits, edge_states, node_states)


class NodeGRU(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, size))
        self.gru = nn.GRUCell(size, size)

    def forward(self, src, dst, logits, nodes):
        weights = segment_softmax(logits, dst, len(nodes))
        context = segment_sum(weights * self.project_node(nodes)[src], dst, len(nodes))
        return F.relu(self.gru(F.elu(context), nodes))


class MessagePassing(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.project_edge = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * size, 1), nn.LeakyReLU())
        self.attentive_gru = NodeGRU(size, dropout)

    def forward(self, src, dst, nodes):
        logits = self.project_edge(torch.cat((nodes[dst], nodes[src]), dim=1))
        return self.attentive_gru(src, dst, logits, nodes)


class AttentiveFPGNN(nn.Module):
    def __init__(self, node_size, edge_size, num_layers, size, dropout):
        super().__init__()
        self.init_context = InitialContext(node_size, edge_size, size, dropout)
        self.gnn_layers = nn.ModuleList(
            MessagePassing(size, dropout) for _ in range(num_layers - 1)
        )

    def forward(self, edge_index, nodes, edges):
        src, dst = edge_index
        nodes = self.init_context(src, dst, nodes, edges)
        for layer in self.gnn_layers:
            nodes = layer(src, dst, nodes)
        return nodes


class GlobalPool(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.compute_logits = nn.Sequential(nn.Linear(2 * size, 1), nn.LeakyReLU())
        self.project_nodes = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, size))
        self.gru = nn.GRUCell(size, size)

    def forward(self, membership, nodes, graphs):
        logits = self.compute_logits(torch.cat((F.relu(graphs)[membership], nodes), dim=1))
        weights = segment_softmax(logits, membership, len(graphs))
        context = segment_sum(weights * self.project_nodes(nodes), membership, len(graphs))
        return self.gru(F.elu(context), graphs), weights


class AttentiveFPReadout(nn.Module):
    def __init__(self, size, num_timesteps, dropout):
        super().__init__()
        self.readouts = nn.ModuleList(GlobalPool(size, dropout) for _ in range(num_timesteps))

    def forward(self, membership, count, nodes, get_node_weight=False):
        graphs = segment_sum(nodes, membership, count)
        node_weights = []
        for readout in self.readouts:
            graphs, weights = readout(membership, nodes, graphs)
            if get_node_weight:
                node_weights.append(weights)
        return (graphs, node_weights) if get_node_weight else graphs
