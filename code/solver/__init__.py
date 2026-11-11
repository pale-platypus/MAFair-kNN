from .ilp_solver import ILPSolver
from .milp_solver import MILPSolver
from .post_alg1_solver import PostAlg1Solver
from .flow_ortools_solver import FlowORToolsSolver
from .greedy_solver import GreedySolver
from .dispatch_solver import DispatchSolver


solvers = {
    "post_alg": DispatchSolver,      # Post-Alg-1 (m=1), Post-Alg-2 (m=2), else the ILP (HiGHS)
    "post_alg1": PostAlg1Solver,     # Post-Alg-1
    "post_alg2": FlowORToolsSolver,  # Post-Alg-2: min-cost flow (OR-Tools)
    "milp": MILPSolver,              # Post-Alg-3: integer program, HiGHS
    "ilp": ILPSolver,                # the same integer program, CBC (PuLP)
    "greedy": GreedySolver,          # greedy selection baseline
}


def build_solver(args):
    if solvers.get(args.solver, None) is not None:
        return solvers[args.solver]()
    else:
        raise NameError(f"Undefined indexing method for {args.solver}!")
