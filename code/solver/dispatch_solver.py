"""
dispatch_solver.py
------------------
The post-processing of the paper's end-to-end algorithms Alg-1-Fair, Alg-2-Fair
and Alg-3+-Fair: Post-Alg-1 when the query constrains one attribute, Post-Alg-2
(min-cost flow) when it constrains two, and Post-Alg-3+ (the integer program,
HiGHS) when it constrains three or more.
"""
from __future__ import annotations
from .post_alg1_solver import PostAlg1Solver
from .flow_ortools_solver import FlowORToolsSolver
from .milp_solver import MILPSolver


class DispatchSolver:
    def __init__(self, msg: bool = False, time_limit=None):
        self.time_limit = time_limit

    def solve(self, candidates, query, value_alias=None):
        m = len(query["count"])
        if m == 1:
            return PostAlg1Solver().solve(candidates, query)
        if m == 2:
            return FlowORToolsSolver().solve(candidates, query)
        return MILPSolver(time_limit=self.time_limit).solve(candidates, query)
