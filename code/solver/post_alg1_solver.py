"""
post_alg1_solver.py
-------------------
Post-Alg-1 (Algorithm 1 of the paper): exact post-processing for one constrained
attribute with ranged constraints [l_v, u_v].

    for each listed value v:
        P_v <- the min(u_v, |C_v|) closest candidates of value v (heap extraction, ascending)
        take the l_v closest items of P_v
    fill the remaining r = k - sum_v l_v slots greedily: a heap holds the next unselected
    item of every P_v; pop the minimum and push its successor, r times.

Candidates whose value is not listed by the query are not admitted (u = 0), as in
MILPSolver. Same result format as MILPSolver.
"""
from __future__ import annotations
import heapq


def _attrs(meta: str) -> dict:
    return dict(p.split(":", 1) for p in meta.split("__") if ":" in p)


def _bounds(c):
    return (int(c[0]), int(c[1])) if isinstance(c, (tuple, list)) else (int(c), int(c))


class PostAlg1Solver:
    def __init__(self, msg: bool = False, time_limit=None):
        pass

    def solve(self, candidates, query, value_alias=None):
        k = int(query["k"])
        qc = query["count"]
        fail = {"status": "Infeasible", "indices": [], "objective": float("inf"),
                "count": {}, "selected": []}
        if len(qc) != 1:
            raise ValueError("Post-Alg-1 handles exactly one constrained attribute")
        (a, vr), = qc.items()
        groups = {v: [] for v in vr}
        for meta, dist in candidates:
            v = _attrs(meta).get(a)
            if v in groups:
                groups[v].append((float(dist), meta))

        sel, resid = [], []
        for v, c in vr.items():
            lb, ub = _bounds(c)
            cv = groups[v]
            heapq.heapify(cv)                                   # O(n_v)
            pv = [heapq.heappop(cv) for _ in range(min(ub, k, len(cv)))]   # ascending
            if len(pv) < lb:
                return fail
            sel.extend(pv[:lb])                                 # mandatory lower bound
            if len(pv) > lb:
                resid.append(pv[lb:])
        r = k - len(sel)
        if r < 0:
            return fail
        h = [(lst[0][0], i, 0) for i, lst in enumerate(resid)]
        heapq.heapify(h)
        for _ in range(r):                                      # greedy fill
            if not h:
                return fail
            _, i, j = heapq.heappop(h)
            sel.append(resid[i][j])
            if j + 1 < len(resid[i]):
                heapq.heappush(h, (resid[i][j + 1][0], i, j + 1))

        counts = {a: {}}
        for _, meta in sel:
            v = _attrs(meta)[a]
            counts[a][v] = counts[a].get(v, 0) + 1
        return {"status": "Optimal", "indices": list(range(len(sel))),
                "objective": float(sum(d for d, _ in sel)),
                "count": counts, "selected": [(meta, d) for d, meta in sel]}
