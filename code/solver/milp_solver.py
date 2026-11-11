"""
milp_solver.py
--------------
Post-Alg-3+ (and Post-Alg-1) as an exact 0/1 integer program, solved in-process
with HiGHS (scipy.optimize.milp). Same model as ilp_solver.py (PuLP + CBC):

    min  sum_i d_i z_i
    s.t. sum_i z_i = k
         l_{j,v} <= sum_{i: a_ij = v} z_i <= u_{j,v}   for every listed (j, v)
         z_i in {0, 1}

Candidates whose value of a constrained attribute is not listed by the query are
not admitted (u = 0) and are dropped before solving. Avoiding CBC's subprocess
start-up removes a ~20 ms constant from every query; the optimum is identical.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


def _attrs(meta: str) -> dict:
    return dict(p.split(":", 1) for p in meta.split("__") if ":" in p)


def _bounds(c):
    return (int(c[0]), int(c[1])) if isinstance(c, (tuple, list)) else (int(c), int(c))


class MILPSolver:
    def __init__(self, msg: bool = False, time_limit=None):
        self.time_limit = time_limit

    def solve(self, candidates, query, value_alias=None):
        k = int(query["k"])
        qc = query["count"]
        metas, dists = [], []
        for meta, dist in candidates:
            md = _attrs(meta)
            if all(md.get(a) in vr for a, vr in qc.items()):
                metas.append((meta, md))
                dists.append(float(dist))
        fail = {"status": "Infeasible", "indices": [], "objective": float("inf"),
                "count": {}, "selected": []}
        n = len(metas)
        if n < k:
            return fail

        rows, cols, lo, hi = [], [], [k], [k]
        rows += [0] * n
        cols += list(range(n))
        r = 1
        for a, vr in qc.items():
            for v, c in vr.items():
                lb, ub = _bounds(c)
                if lb <= 0 and ub >= k:
                    continue
                idx = [i for i, (_, md) in enumerate(metas) if md[a] == v]
                rows += [r] * len(idx)
                cols += idx
                lo.append(lb)
                hi.append(min(ub, k))
                r += 1
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, n))
        opts = {"disp": False}
        if self.time_limit:
            opts["time_limit"] = self.time_limit
        res = milp(c=np.asarray(dists), constraints=LinearConstraint(A, lo, hi),
                   integrality=np.ones(n), bounds=Bounds(0, 1), options=opts)
        if res.status != 0 or res.x is None:
            return fail
        sel = [i for i in range(n) if res.x[i] > 0.5]
        if len(sel) != k:
            return fail
        selected = [(metas[i][0], dists[i]) for i in sel]
        counts = {a: {} for a in qc}
        for i in sel:
            for a in qc:
                v = metas[i][1][a]
                counts[a][v] = counts[a].get(v, 0) + 1
        return {"status": "Optimal", "indices": sel,
                "objective": float(sum(dists[i] for i in sel)),
                "count": counts, "selected": selected}
