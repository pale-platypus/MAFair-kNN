"""
flow_ortools_solver.py
----------------------
Post-Alg-2 (Section III-A2 of the paper) on a compiled min-cost flow solver
(OR-Tools SimpleMinCostFlow, C++):

    Source -> U1,v      capacity l_{1,v}
    Source -> R_src     capacity a  = min(k, sum_v u_{1,v}) - sum_v l_{1,v}
    R_src  -> U1,v      capacity u_{1,v} - l_{1,v}
    U1,v   -> U2,w      one arc per candidate (v, w), capacity 1, cost = its distance
    U2,w   -> Sink      capacity l_{2,w}
    U2,w   -> R_snk     capacity u_{2,w} - l_{2,w}
    R_snk  -> Sink      capacity a' = min(k, sum_w u_{2,w}) - sum_w l_{2,w}

Source supplies k and Sink absorbs k, so a feasible flow of value k is a fair
k-subset of the pool and a minimum-cost one is an optimal fair subset. Costs are
distances scaled to integers, as in the NetworkX version. Candidates whose value
of a constrained attribute is not listed get no arc (not admitted). Same result
format as MILPSolver.
"""
from __future__ import annotations
import numpy as np
from ortools.graph.python import min_cost_flow


def _attrs(meta: str) -> dict:
    return dict(p.split(":", 1) for p in meta.split("__") if ":" in p)


def _bounds(c):
    return (int(c[0]), int(c[1])) if isinstance(c, (tuple, list)) else (int(c), int(c))


class FlowORToolsSolver:
    def __init__(self, msg: bool = False, time_limit=None):
        pass

    def solve(self, candidates, query, value_alias=None):
        k = int(query["k"])
        qc = query["count"]
        fail = {"status": "Infeasible", "indices": [], "objective": float("inf"),
                "count": {}, "selected": []}
        if len(qc) != 2:
            raise ValueError("Post-Alg-2 handles exactly two constrained attributes")
        (a1, vr1), (a2, vr2) = qc.items()
        b1 = {v: _bounds(c) for v, c in vr1.items()}
        b2 = {w: _bounds(c) for w, c in vr2.items()}

        SRC, SNK, RSRC, RSNK = 0, 1, 2, 3
        n1 = {v: 4 + i for i, v in enumerate(b1)}
        n2 = {w: 4 + len(b1) + i for i, w in enumerate(b2)}
        tails, heads, caps, costs = [], [], [], []

        def arc(t, h, cap, cost=0):
            if cap > 0:
                tails.append(t); heads.append(h); caps.append(cap); costs.append(cost)

        arc(SRC, RSRC, min(k, sum(u for _, u in b1.values())) - sum(l for l, _ in b1.values()))
        for v, (l, u) in b1.items():
            arc(SRC, n1[v], l)
            arc(RSRC, n1[v], min(u, k) - l)
        arc(RSNK, SNK, min(k, sum(u for _, u in b2.values())) - sum(l for l, _ in b2.values()))
        for w, (l, u) in b2.items():
            arc(n2[w], SNK, l)
            arc(n2[w], RSNK, min(u, k) - l)
        n_infra = len(tails)

        admitted = []
        for meta, dist in candidates:
            md = _attrs(meta)
            v, w = md.get(a1), md.get(a2)
            if v in n1 and w in n2:
                admitted.append((meta, float(dist), md))
        if len(admitted) < k:
            return fail
        scale = 1e6 / max(1e-12, max(d for _, d, _ in admitted))
        for meta, dist, md in admitted:
            tails.append(n1[md[a1]]); heads.append(n2[md[a2]])
            caps.append(1); costs.append(int(round(dist * scale)))

        smcf = min_cost_flow.SimpleMinCostFlow()
        arcs = smcf.add_arcs_with_capacity_and_unit_cost(
            np.asarray(tails, dtype=np.int64), np.asarray(heads, dtype=np.int64),
            np.asarray(caps, dtype=np.int64), np.asarray(costs, dtype=np.int64))
        smcf.set_node_supply(SRC, k)
        smcf.set_node_supply(SNK, -k)
        if smcf.solve() != smcf.OPTIMAL:
            return fail
        flows = smcf.flows(arcs[n_infra:])
        sel = [i for i in range(len(admitted)) if flows[i] > 0]
        if len(sel) != k:
            return fail
        selected = [(admitted[i][0], admitted[i][1]) for i in sel]
        counts = {a1: {}, a2: {}}
        for i in sel:
            md = admitted[i][2]
            for a in (a1, a2):
                counts[a][md[a]] = counts[a].get(md[a], 0) + 1
        return {"status": "Optimal", "indices": sel,
                "objective": float(sum(d for _, d in selected)),
                "count": counts, "selected": selected}
