"""Small LSTM and LSTM-GNN forecasting models with controlled graph updates.

Adjacency entry A[i, j] is the weight of the message sent from node j to node i.
Graph construction uses only the supplied observed-history input tensor.
"""

import operator

import torch
from torch import nn


def _history_batch(x):
    """Return a floating [batch, history, nodes, 1] tensor and a squeeze flag."""
    x = torch.as_tensor(x)
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[-1] != 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("Expected [batch, history, nodes, 1] or [history, nodes, 1]")
    if not x.is_floating_point():
        x = x.to(dtype=torch.get_default_dtype())
    if not torch.isfinite(x).all():
        raise ValueError("Graph inputs must be finite after the common input-imputation rule")
    return x, squeeze


def _absolute_correlations(x, epsilon=1e-12):
    """Return dense absolute Pearson correlations, with constant nodes disconnected."""
    series = x[..., 0]
    centered = series - series.mean(dim=1, keepdim=True)
    products = torch.bmm(centered.transpose(1, 2), centered)
    norms = torch.linalg.vector_norm(centered, dim=1)
    denominator = norms.unsqueeze(2) * norms.unsqueeze(1)
    correlations = (products / denominator.clamp_min(epsilon)).abs().clamp(max=1)
    varying = series.amax(dim=1) > series.amin(dim=1)
    valid_pairs = varying.unsqueeze(2) & varying.unsqueeze(1)
    n_nodes = series.shape[2]
    diagonal = torch.eye(n_nodes, dtype=torch.bool, device=x.device).unsqueeze(0)
    return correlations.masked_fill(~valid_pairs | diagonal, 0)


def _row_normalize(weights):
    """Preserve an all-zero row instead of manufacturing edges for a constant node."""
    sums = weights.sum(dim=-1, keepdim=True)
    return torch.where(sums > 0, weights / sums.clamp_min(torch.finfo(weights.dtype).tiny),
                       torch.zeros_like(weights))


def build_adjacency(x, top_k=3):
    """Build a directed row-normalized top-k absolute-correlation graph.

    Equal correlations prefer the smaller source-node index. Self-edges are
    excluded. Zero-weight candidates never become effective edges, so a node
    may have fewer than k neighbors. An unbatched history returns [N, N].
    """
    try:
        k = operator.index(top_k)
    except TypeError as exc:
        raise ValueError("top_k must be an integer") from exc
    if k < 0:
        raise ValueError("top_k must be nonnegative")
    x, squeeze = _history_batch(x)
    correlations = _absolute_correlations(x)
    n_nodes = x.shape[2]
    k = min(k, n_nodes - 1)
    if k:
        diagonal = torch.eye(n_nodes, dtype=torch.bool, device=x.device).unsqueeze(0)
        ranking = correlations.masked_fill(diagonal, -torch.inf)
        ordered = torch.argsort(ranking, dim=-1, descending=True, stable=True)
        mask = torch.zeros_like(correlations, dtype=torch.bool)
        mask.scatter_(-1, ordered[..., :k], True)
        weights = correlations.masked_fill(~mask, 0)
    else:
        weights = torch.zeros_like(correlations)
    adjacency = _row_normalize(weights)
    return adjacency[0] if squeeze else adjacency


def _validate_model_input(x, n_nodes):
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[-1] != 1:
        raise ValueError("Model input must be a tensor shaped [batch, history, nodes, 1]")
    if x.shape[2] != n_nodes or x.shape[1] < 1:
        raise ValueError("Input node count/history does not match the model")
    if not x.is_floating_point():
        raise ValueError("Model input must have a floating dtype")


class MultivariateLSTM(nn.Module):
    """A graph-free LSTM that jointly observes every node's scalar time series."""

    def __init__(self, n_nodes, hidden=16, layers=1, dropout=0.0):
        super().__init__()
        if n_nodes < 1 or hidden < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LSTM dimensions or dropout")
        self.n_nodes = int(n_nodes)
        self.temporal = nn.LSTM(self.n_nodes, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.readout = nn.Linear(hidden, self.n_nodes)

    def forward(self, x):
        _validate_model_input(x, self.n_nodes)
        _, (hidden, _) = self.temporal(x[..., 0])
        return self.readout(hidden[-1])


class LSTMGNN(nn.Module):
    """Shared per-node LSTM followed by one weighted message-passing layer.

    Static and dynamic modes use exactly the same learned parameters. Graph
    construction is a deterministic input transformation, not a learned edge
    mechanism. Static adjacency is a buffer and is always present in state_dict.
    """

    MODES = ("static", "dynamic", "fixed_edges_dynamic_weights")

    def __init__(self, n_nodes, hidden=16, top_k=3, graph_mode="static",
                 static_adjacency=None, layers=1, dropout=0.0):
        super().__init__()
        if graph_mode not in self.MODES:
            raise ValueError("Unknown graph_mode: " + str(graph_mode))
        if n_nodes < 1 or hidden < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LSTM-GNN dimensions or dropout")
        try:
            k = operator.index(top_k)
        except TypeError as exc:
            raise ValueError("top_k must be an integer") from exc
        if k < 0:
            raise ValueError("top_k must be nonnegative")
        self.n_nodes = int(n_nodes)
        self.top_k = k
        self.graph_mode = graph_mode
        if static_adjacency is None:
            if graph_mode != "dynamic":
                raise ValueError("A training-period static_adjacency is required for this mode")
            adjacency = torch.zeros(self.n_nodes, self.n_nodes)
        else:
            adjacency = torch.as_tensor(static_adjacency, dtype=torch.float32).detach().clone()
            if adjacency.shape != (self.n_nodes, self.n_nodes):
                raise ValueError("static_adjacency must have shape [nodes, nodes]")
            if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
                raise ValueError("static_adjacency must be finite and nonnegative")
            adjacency.diagonal().zero_()
            adjacency = _row_normalize(adjacency)
        self.register_buffer("static_adjacency", adjacency)
        self.temporal = nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.self_projection = nn.Linear(hidden, hidden)
        self.neighbor_projection = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, 1)

    def graph_adjacency(self, x):
        """Return the graph used for this exact history batch, without future labels."""
        _validate_model_input(x, self.n_nodes)
        if self.graph_mode == "static":
            return self.static_adjacency.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.graph_mode == "dynamic":
            return build_adjacency(x, self.top_k)
        history, _ = _history_batch(x)
        weights = _absolute_correlations(history)
        weights = weights * (self.static_adjacency > 0).unsqueeze(0)
        return _row_normalize(weights)

    def forward(self, x):
        _validate_model_input(x, self.n_nodes)
        batch, length, n_nodes, _ = x.shape
        sequences = x.permute(0, 2, 1, 3).reshape(batch * n_nodes, length, 1)
        _, (hidden, _) = self.temporal(sequences)
        states = hidden[-1].reshape(batch, n_nodes, -1)
        adjacency = self.graph_adjacency(x)
        neighbors = torch.bmm(adjacency, states)
        updated = torch.relu(self.self_projection(states) + self.neighbor_projection(neighbors))
        return self.readout(self.dropout(updated)).squeeze(-1)
