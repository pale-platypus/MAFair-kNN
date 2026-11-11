"""
depth_analysis.py — how deep must a GLOBAL nearest-neighbour ranking go before it
contains the optimal fair answer?  For every (valid) query: rank all points by
distance to q (squared L2, as everywhere else) and report the rank of the
farthest member of the exact fair optimum S*. A global index that retrieves
the top-K and post-filters can return S* only if K >= that rank.

Writes results/depth.csv (one row per workload).
"""
import os, pickle, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate_ranged import query_is_valid, extract_id  # noqa: E402

WORKLOADS = [("fairface", "k=10_m=3"), ("audio", "k=10_m=3"), ("celeba", "k=10_m=1"), ("celeba", "k=10_m=2"),
             ("celeba", "k=10_m=3"), ("celeba", "k=10_m=4"), ("celeba", "k=10_m=5"), ("sift1m", "k=10_m=3")]

rows = []
for ds, tag in WORKLOADS:
    qp = os.path.join(ROOT, "data", ds, "icde", f"ranged_queries_{tag}_fdist=euclidean_200.pkl")
    if not os.path.exists(qp):
        print("missing", qp); continue
    X = np.load(os.path.join(ROOT, "data", ds, "vectors.npz"))["vectors"].astype(np.float32)
    sq = np.einsum("ij,ij->i", X, X)
    Q = [q for q in pickle.load(open(qp, "rb")) if query_is_valid(q)]
    depth = []
    for q in Q:
        v = np.asarray(q["vector"], dtype=np.float32)
        d = sq - 2 * (X @ v) + v @ v
        ids = [extract_id(m) for m in q["ground_truth"]]
        dmax = d[ids].max()
        depth.append(int((d <= dmax).sum()))
    r = np.array(depth)
    rows.append({"dataset": ds, "tag": tag, "n": len(X), "queries": len(r),
                 "median": int(np.median(r)), "p90": int(np.percentile(r, 90)), "max": int(r.max()),
                 "median_frac": float(np.median(r) / len(X)), "p90_frac": float(np.percentile(r, 90) / len(X)),
                 "frac_K<=1000": float(np.mean(r <= 1000)), "frac_K<=10000": float(np.mean(r <= 10000))})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "depth.csv"), index=False)
