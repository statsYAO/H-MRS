
import numpy as np
from typing import Tuple, List, Dict


def generate_log_linear_data(p: int, d: int, n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Generate synthetic data from the log-linear model."""
    np.random.seed(seed)

    ordering = np.random.permutation(p)
    adj_matrix = np.zeros((p, p), dtype=int)
    true_edges = []

    for i, node in enumerate(ordering):
        if i == 0:
            continue

        possible_parents = ordering[:i]
        num_parents = np.random.randint(0, min(d + 1, len(possible_parents) + 1))

        if num_parents > 0:
            parents = np.random.choice(possible_parents, num_parents, replace=False)
            for parent in parents:
                adj_matrix[parent, node] = 1
                true_edges.append((parent, node))

    intercepts = np.random.uniform(0.5, 2.0, p)
    noise_std = np.random.uniform(0.1, 0.3, p)

    edge_weights = {}
    for parent, child in true_edges:
        edge_weights[(parent, child)] = np.random.uniform(-0.3, 0.3)

    X = np.zeros((n, p))

    for i in range(n):
        for node in ordering:
            log_mean = intercepts[node]

            parents = np.where(adj_matrix[:, node] == 1)[0]
            for parent in parents:
                log_mean += edge_weights[(parent, node)] * X[i, parent]

            noise = np.random.uniform(-0.5, 0.5)
            log_X = log_mean + noise_std[node] * noise
            X[i, node] = np.exp(log_X)

    return X, adj_matrix, true_edges


def compute_shd(true_adj: np.ndarray, estimated_adj: np.ndarray) -> int:
    """Compute Structural Hamming Distance."""
    true_binary = (true_adj != 0).astype(int)
    est_binary = (estimated_adj != 0).astype(int)

    missing_edges = np.sum((true_binary == 1) & (est_binary == 0))
    extra_edges = np.sum((true_binary == 0) & (est_binary == 1))

    reversed_edges = 0
    for i in range(true_adj.shape[0]):
        for j in range(true_adj.shape[1]):
            if true_binary[i, j] == 1 and est_binary[i, j] == 0 and est_binary[j, i] == 1:
                reversed_edges += 1

    return missing_edges + extra_edges + reversed_edges


def evaluate_performance_with_shd(true_edges: List[Tuple[int, int]],
                                  estimated_adj: np.ndarray,
                                  true_adj: np.ndarray) -> Dict[str, float]:
    """Evaluate graph recovery performance."""
    estimated_edges = []
    for i in range(estimated_adj.shape[0]):
        for j in range(estimated_adj.shape[1]):
            if estimated_adj[i, j] == 1:
                estimated_edges.append((i, j))

    true_set = set(true_edges)
    est_set = set(estimated_edges)

    if len(est_set) == 0:
        precision = 0.0 if len(true_set) > 0 else 1.0
    else:
        precision = len(true_set.intersection(est_set)) / len(est_set)

    if len(true_set) == 0:
        recall = 1.0 if len(est_set) == 0 else 0.0
    else:
        recall = len(true_set.intersection(est_set)) / len(true_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    shd = compute_shd(true_adj, estimated_adj)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "true_edges": len(true_set),
        "estimated_edges": len(est_set)
    }
