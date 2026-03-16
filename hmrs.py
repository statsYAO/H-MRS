
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from typing import Tuple, List, Dict


class LogLinearMRS:
    """Hybrid Moment-Ratio Scoring for positive-valued DAG learning."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

        self.ridge_alpha = 0.01
        self.elasticnet_alpha = 0.1
        self.l1_ratio = 0.8
        self.threshold = 0.01
        self.max_parents = 2
        self.tie_breaking_weight = 1e-12

    def compute_moment_score(self, X: np.ndarray, j: int, S: List[int]) -> float:
        """Compute the moment-ratio score."""
        E_X2 = np.mean(X[:, j] ** 2)

        if len(S) == 0:
            E_X = np.mean(X[:, j])
            denominator = E_X ** 2
        else:
            X_parents = X[:, S]
            log_y = np.log(np.maximum(X[:, j], 1e-10))

            try:
                reg = Ridge(alpha=self.ridge_alpha)

                if X_parents.ndim == 1:
                    X_parents = X_parents.reshape(-1, 1)

                reg.fit(X_parents, log_y)
                log_pred = reg.predict(X_parents)

                mu_pred = np.exp(log_pred)
                mu_pred = np.clip(mu_pred, 1e-10, 1e10)

                denominator = np.mean(mu_pred ** 2)

            except Exception:
                E_X = np.mean(X[:, j])
                denominator = E_X ** 2

        denominator = max(denominator, 1e-10)
        score = E_X2 / denominator
        score += j * self.tie_breaking_weight

        return score

    def select_parents(self, X: np.ndarray, j: int, candidates: List[int]) -> List[int]:
        """Select parents using ElasticNet."""
        if len(candidates) == 0:
            return []

        X_candidates = X[:, candidates]
        log_y = np.log(np.maximum(X[:, j], 1e-10))

        try:
            reg = ElasticNet(
                alpha=self.elasticnet_alpha,
                l1_ratio=self.l1_ratio,
                max_iter=2000
            )
            reg.fit(X_candidates, log_y)
            coeffs = reg.coef_

            selected_idx = np.where(np.abs(coeffs) > self.threshold)[0]

            if len(selected_idx) > self.max_parents:
                sorted_idx = np.argsort(np.abs(coeffs))[::-1]
                selected_idx = sorted_idx[:self.max_parents]

            parents = [candidates[i] for i in selected_idx]
            return parents

        except Exception:
            return []

    def fit(self, X: np.ndarray) -> Tuple[List[int], Dict[int, List[int]]]:
        """Learn the DAG structure."""
        n, p = X.shape

        ordering = []
        parents = {}
        remaining = list(range(p))

        for step in range(p):
            scores = {}

            for j in remaining:
                if step == 0:
                    score = self.compute_moment_score(X, j, [])
                else:
                    score = self.compute_moment_score(X, j, ordering)

                scores[j] = score

            next_node = min(scores.keys(), key=lambda x: scores[x])
            ordering.append(next_node)
            remaining.remove(next_node)

            if step == 0:
                parents[next_node] = []
            else:
                node_parents = self.select_parents(X, next_node, ordering[:-1])
                parents[next_node] = node_parents

        return ordering, parents
