"""
check_post.py - per query, compare a post-processing solver with the HiGHS integer
program (milp) on the same exact pools (index exact_cartesian):

  * exact solvers (post_alg1, post_alg2, ilp, post_alg): number of
    queries whose objective differs from the ILP optimum (should be 0);
  * greedy: share of queries it fails on although the ILP finds a fair answer in the pool,
    and its objective relative to the optimum where it succeeds.

    python scripts/check_post.py SOLVER DATASET:TAG [DATASET:TAG ...]
"""
import os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate_ranged import strict_feasible  # noqa: E402


def load(ds, tag, solver):
    p = os.path.join(ROOT, "outputs", ds, f"results_ranged_{tag}", "exact_cartesian", f"exact__solver={solver}.pkl")
    return pickle.load(open(p, "rb"))["query_results"] if os.path.exists(p) else None


def queries(ds, tag):
    return pickle.load(open(os.path.join(ROOT, "data", ds, "icde", f"ranged_queries_{tag}_fdist=euclidean_200.pkl"), "rb"))


def obj(r, q):
    if r is None or not r.get("selected") or not strict_feasible(r["selected"], q):
        return None
    return sum(float(m[1]) for m in r["selected"])


def ms(R, key="postprocessing_time"):
    v = [r[key] for r in R if r is not None and r.get(key) is not None]
    return 1e3 * float(np.mean(v)) if v else float("nan")


def main():
    solver, specs = sys.argv[1], sys.argv[2:]
    for spec in specs:
        ds, tag = spec.split(":", 1)
        S, B = load(ds, tag, solver), load(ds, tag, "milp")
        if S is None or B is None:
            print(f"{ds:9s} {tag:12s} missing {solver if S is None else 'milp'}"); continue
        Q = queries(ds, tag)
        n = min(len(S), len(B), len(Q))
        opt = [obj(B[i], Q[i]) for i in range(n)]
        got = [obj(S[i], Q[i]) for i in range(n)]
        has = [i for i in range(n) if opt[i] is not None]          # pool holds a fair answer
        fails = sum(1 for i in has if got[i] is None)
        ratios = [got[i] / opt[i] for i in has if got[i] is not None and opt[i] > 0]
        worse = sum(1 for r in ratios if r > 1 + 1e-6)
        mism = sum(1 for i in range(n) if (got[i] is None) != (opt[i] is None)
                   or (got[i] is not None and abs(got[i] - opt[i]) > 1e-6 * max(1.0, abs(opt[i]))))
        pool = np.mean([r.get("pool_size", np.nan) for r in S if r is not None])
        print(f"{ds:9s} {tag:12s} n={n} fair_pools={len(has)} mismatches={mism} fails={fails} "
              f"({100 * fails / max(1, len(has)):.1f}%) costlier={worse} "
              f"mean_ratio={np.mean(ratios) if ratios else float('nan'):.4f} "
              f"max_ratio={max(ratios) if ratios else float('nan'):.4f} "
              f"{solver}={ms(S):.3f}ms milp={ms(B):.3f}ms pool={pool:.1f}")


if __name__ == "__main__":
    main()
