
import numpy as np
from hmrs import LogLinearMRS
from simulate_data import generate_log_linear_data, evaluate_performance_with_shd


params = {
    "ridge_alpha": 0.1,
    "elasticnet_alpha": 0.01,
    "l1_ratio": 0.9,
    "threshold": 0.05,
    "max_parents": 2
}

X, true_adj, true_edges = generate_log_linear_data(p=10, d=1, n=500, seed=42)

model = LogLinearMRS(random_state=42)

model.ridge_alpha = params["ridge_alpha"]
model.elasticnet_alpha = params["elasticnet_alpha"]
model.l1_ratio = params["l1_ratio"]
model.threshold = params["threshold"]
model.max_parents = params["max_parents"]

ordering, parents = model.fit(X)

p = X.shape[1]
estimated_adj = np.zeros((p, p), dtype=int)

for child, parent_list in parents.items():
    for parent in parent_list:
        estimated_adj[parent, child] = 1

metrics = evaluate_performance_with_shd(true_edges, estimated_adj, true_adj)

print("Estimated ordering:", ordering)
print("Estimated parents:", parents)
print("Performance metrics:", metrics)
