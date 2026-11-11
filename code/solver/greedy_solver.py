"""
greedy_solver.py
----------------
Greedy fair selection, the post-processing baseline. It scans the pool by
increasing distance and adds a candidate unless doing so would

    (i)  exceed an upper bound u_{j,v} of one of its values, or
    (ii) leave fewer free slots than some attribute still needs to reach its
         lower bounds: k - |S| < sum_v max(0, l_{j,v} - |S_{j,v}|) for some A_j.

Each check concerns one attribute at a time, so the greedy enforces every
attribute's bounds separately. For one attribute this is the greedy algorithm on
a matroid and returns the optimum (as Post-Alg-1 does); for two or more it can
return a costlier answer or none, even when the pool contains a fair answer.
Candidates whose value of a constrained attribute is not listed are not admitted.
Same result format as MILPSolver.
"""
from __future__ import annotations


def _attrs(meta: str) -> dict:
    return dict(p.split(":", 1) for p in meta.split("__") if ":" in p)


def _bounds(c):
    return (int(c[0]), int(c[1])) if isinstance(c, (tuple, list)) else (int(c), int(c))


class GreedySolver:
    def __init__(self, msg: bool = False, time_limit=None):
        pass

    def solve(self, candidates, query, value_alias=None):
        k = int(query["k"])
        qc = query["count"]
        fail = {"status": "Infeasible", "indices": [], "objective": float("inf"),
                "count": {}, "selected": []}
        bnd = {a: {v: _bounds(c) for v, c in vr.items()} for a, vr in qc.items()}
        cnt = {a: {v: 0 for v in vr} for a, vr in bnd.items()}
        need = {a: sum(l for l, _ in vr.values()) for a, vr in bnd.items()}   # unmet lower bounds

        pool = []
        for meta, dist in candidates:
            md = _attrs(meta)
            if all(md.get(a) in vr for a, vr in bnd.items()):
                pool.append((float(dist), meta, md))
        pool.sort(key=lambda t: t[0])

        sel = []
        for dist, meta, md in pool:
            if len(sel) == k:
                break
            if any(cnt[a][md[a]] >= bnd[a][md[a]][1] for a in bnd):
                continue
            free = k - len(sel) - 1
            if any(need[a] - (cnt[a][md[a]] < bnd[a][md[a]][0]) > free for a in bnd):
                continue
            sel.append((meta, dist))
            for a in bnd:
                if cnt[a][md[a]] < bnd[a][md[a]][0]:
                    need[a] -= 1
                cnt[a][md[a]] += 1
        if len(sel) != k or any(need[a] > 0 for a in bnd):
            return fail
        counts = {a: {v: c for v, c in cv.items() if c} for a, cv in cnt.items()}
        return {"status": "Optimal", "indices": list(range(k)),
                "objective": float(sum(d for _, d in sel)),
                "count": counts, "selected": sel}
