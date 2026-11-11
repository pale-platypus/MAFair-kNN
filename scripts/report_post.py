"""
report_post.py - the numbers of Section V-B (post-processing on exact pools), read from
outputs/<dataset>/results_ranged_<tag>/exact_cartesian/exact__solver=<solver>.pkl.

    python scripts/report_post.py

Per workload: mean post-processing time of every solver, mean pool size, the speed-up of
Post-Alg-1 / Post-Alg-2 over HiGHS, objective mismatches against HiGHS (must be 0 for the
exact solvers), and for greedy selection the share of queries it fails on although the
pool holds a fair answer and its objective relative to the optimum where it succeeds.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_post import load, queries, obj  # noqa: E402


def ms(R):
    v = [r["postprocessing_time"] for r in R if r is not None and r.get("postprocessing_time") is not None]
    return 1e3 * float(np.mean(v)) if v else float("nan")


def pool(R):
    v = [r["pool_size"] for r in R if r is not None and r.get("pool_size") is not None]
    return float(np.mean(v)) if v else float("nan")


def compare(S, B, Q):
    n = min(len(S), len(B), len(Q))
    opt = [obj(B[i], Q[i]) for i in range(n)]
    got = [obj(S[i], Q[i]) for i in range(n)]
    has = [i for i in range(n) if opt[i] is not None]
    mism = sum(1 for i in range(n) if (got[i] is None) != (opt[i] is None)
               or (got[i] is not None and abs(got[i] - opt[i]) > 1e-6 * max(1.0, abs(opt[i]))))
    fails = sum(1 for i in has if got[i] is None)
    ratios = [got[i] / opt[i] for i in has if got[i] is not None and opt[i] > 0]
    costlier = sum(1 for r in ratios if r > 1 + 1e-6)
    return dict(n=n, fair=len(has), mism=mism, fail_pct=100.0 * fails / max(1, len(has)),
                costlier_pct=100.0 * costlier / max(1, len(has)),
                mean_excess_pct=100.0 * (np.mean(ratios) - 1) if ratios else float("nan"),
                max_excess_pct=100.0 * (max(ratios) - 1) if ratios else float("nan"))


def row(ds, tag, solvers):
    Q = queries(ds, tag)
    R = {s: load(ds, tag, s) for s in solvers + ["milp"]}
    B = R["milp"]
    out = [f"{ds:9s} {tag:13s} pool={pool(B):7.1f}"]
    for s in solvers + ["milp"]:
        if R[s] is None:
            out.append(f"{s}=missing"); continue
        out.append(f"{s}={ms(R[s]):8.3f}ms")
        if s != "milp":
            c = compare(R[s], B, Q)
            if s == "greedy":
                out.append(f"[greedy fail={c['fail_pct']:5.1f}% costlier={c['costlier_pct']:5.1f}% "
                           f"excess mean={c['mean_excess_pct']:.2f}% max={c['max_excess_pct']:.2f}% fair={c['fair']}]")
            else:
                out.append(f"[mism={c['mism']} x{ms(B) / ms(R[s]):.1f}]")
    print("  ".join(out))


def main():
    print("== m=1 / m=2 on FairFace, varying k")
    for m, alg in ((1, "post_alg1"), (2, "post_alg2")):
        for k in (10, 20, 50, 100, 200, 500, 1000):
            row("fairface", f"k={k}_m={m}", [alg, "ilp", "greedy"])
    print("== m=3 on FairFace, varying k")
    for k in (5, 10, 15, 20, 50, 100, 200, 500, 1000):
        row("fairface", f"k={k}_m=3", ["ilp", "greedy"])
    print("== CelebA-B (cbin), varying m")
    for m in (2, 3, 4, 5, 6, 8, 10):
        row("cbin", f"k=10_m={m}", ["ilp", "greedy"] + (["post_alg2"] if m == 2 else []))
    print("== CelebA, varying m")
    for m in (1, 2, 3, 4, 5):
        row("celeba", f"k=10_m={m}", ["greedy"] + (["post_alg1"] if m == 1 else []))
    print("== Audio and SIFT1M")
    for k in (5, 10, 15, 20):
        row("audio", f"k={k}_m=3", ["greedy"])
    row("sift1m", "k=10_m=3", ["greedy"])


if __name__ == "__main__":
    main()
