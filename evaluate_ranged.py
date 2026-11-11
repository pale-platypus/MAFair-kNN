"""
evaluate_ranged.py
------------------
Parse result pkl files and compute recall@k and percentage of successfully
solved queries for ranged fairness constraints.

Directory layout assumed:
    results_ranged_k={k}_m={m}_fdist={fdist}_{n}/
        {method}/
            {param_string}.pkl     ← list of per-query result dicts

Query files assumed at:
    data/{dataset}/ranged_queries_k={k}_m={m}_fdist={fdist}_{n}.pkl

Usage:
    python evaluate_ranged.py \
        --results_dir ./results_ranged_k=10_m=2_fdist=euclidean_200 \
        --queries_path ./data/celeba/ranged_queries_k=10_m=2_fdist=euclidean_200.pkl \
        --output summary.csv
"""

import os
import pickle
import argparse
import pandas as pd
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_id(meta_str):
    # result["selected"] may contain (meta_str, dist) tuples or plain strings
    if isinstance(meta_str, tuple):
        meta_str = meta_str[0]
    return int(meta_str.split("__")[0].split(":")[1])


def ground_truth_ids(query: dict) -> set[int]:
    """
    query['ground_truth'] is a list of (meta_str, dist) tuples.
    Returns the set of integer ids.
    """
    return {extract_id(m) for m, _ in query["ground_truth"]}


def selected_ids(result: dict) -> set[int]:
    """result['selected'] is a list of meta strings or (meta, dist) tuples."""
    return {extract_id(m) for m in result["selected"]}


def _meta_of(item) -> str:
    return item[0] if isinstance(item, (tuple, list)) else item


def _attrs(meta: str) -> dict:
    return dict(p.split(":", 1) for p in meta.split("__") if ":" in p)


def strict_feasible(selected, query: dict) -> bool:
    """
    STRICT feasibility of a returned set, recomputed from the returned items:
      * exactly k distinct items,
      * every constrained attribute: lb <= count(value) <= ub for each listed value,
      * no item carries a value of a constrained attribute that the query does not
        list (unlisted values are not admitted, i.e. u = 0).
    Self-reported counts are NOT trusted.
    """
    if not selected:
        return False
    ids = [extract_id(m) for m in selected]
    if len(set(ids)) != len(ids) or len(ids) != int(query["k"]):
        return False
    counts = {a: {} for a in query["count"]}
    for item in selected:
        md = _attrs(_meta_of(item))
        for a, vr in query["count"].items():
            v = md.get(a)
            if v not in vr:
                return False
            counts[a][v] = counts[a].get(v, 0) + 1
    for a, vr in query["count"].items():
        for v, c in vr.items():
            lb, ub = (c if isinstance(c, (tuple, list)) else (int(c), int(c)))
            if not (lb <= counts[a].get(v, 0) <= ub):
                return False
    return True


def is_feasible(result: dict, query: dict) -> bool:
    """Strict feasibility of a method's answer (see strict_feasible)."""
    return bool(result) and strict_feasible(result.get("selected"), query)


def query_is_valid(query: dict) -> bool:
    """A query is evaluated only if its stored ground truth is itself a feasible set."""
    return "ground_truth" not in query or strict_feasible(query["ground_truth"], query)


def recall_at_k(result: dict, query: dict):
    """Fraction of ground-truth ids recovered in the result.
    Returns None when the query carries no ground truth (e.g. the 1B run,
    where exact ground truth over 1e9 vectors is infeasible)."""
    if "ground_truth" not in query:
        return None
    k = int(query["k"])
    if not result.get("selected"):
        return 0.0
    gt = ground_truth_ids(query)
    sel = selected_ids(result)
    return len(gt & sel) / k


def _ci95(xs):
    import math
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def _wilson95(succ, n):
    """Half-width of the 95% Wilson interval for a proportion (in %)."""
    import math
    if n == 0:
        return float("nan")
    z, ph = 1.96, succ / n
    den = 1 + z * z / n
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return 100.0 * half


def load_result_file(result_path: str):
    """Return (wrapper_dict_or_None, list_of_per_query_results)."""
    with open(result_path, "rb") as f:
        results = pickle.load(f)
    wrapper = None
    if isinstance(results, dict):
        if "query_results" in results:
            wrapper = results
            results = results["query_results"]
        else:
            results = [results[k] for k in sorted(results.keys(), key=lambda x: int(x))]
    return wrapper, results


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_file(result_path: str, queries: list[dict]) -> dict:
    """
    Strict per-query metrics over the VALID queries only (queries whose stored
    ground truth is a feasible size-k set). Recall@k and success are averaged
    over all valid queries (a failed query has recall 0); DAF is averaged over
    successful queries. 95% confidence half-widths are reported as *_ci.
    """
    wrapper, results = load_result_file(result_path)
    n = min(len(results), len(queries))
    has_ground_truth = bool(queries) and ("ground_truth" in queries[0])

    recalls, dafs, ok_flags = [], [], []
    search_times, post_times, scanned, pools = [], [], [], []
    n_valid = 0
    for i in range(n):
        q = queries[i]
        if not query_is_valid(q):
            continue
        n_valid += 1
        res = results[i]
        if res:
            search_times.append(res.get("search_time", 0.0))
            post_times.append(res.get("postprocessing_time", 0.0))
            scanned.append(res.get("total_scanned", 0))
            pools.append(res.get("pool_size", float("nan")))
        ok = is_feasible(res, q)
        ok_flags.append(ok)
        if has_ground_truth:
            recalls.append(recall_at_k(res, q) if (res and ok) else 0.0)
        gt_dist = q.get("ground_truth_dist")
        if ok and gt_dist and gt_dist > 0:
            obj = sum(float(m[1]) for m in res["selected"]) if isinstance(res["selected"][0], (tuple, list))                 else res.get("objective")
            dafs.append(obj / gt_dist)

    succ = sum(ok_flags)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    out = {
        "n_queries":        n,
        "n_valid":          n_valid,
        "recall@k":         mean(recalls),
        "recall_ci":        _ci95(recalls),
        "DAF":              mean(dafs),
        "DAF_ci":           _ci95(dafs),
        "success_%":        100.0 * succ / n_valid if n_valid else 0.0,
        "success_ci":       _wilson95(succ, n_valid),
        "avg_search_ms":    1000 * mean(search_times) if search_times else float("nan"),
        "avg_post_ms":      1000 * mean(post_times) if post_times else float("nan"),
        "avg_total_ms":     1000 * (mean(search_times) + mean(post_times)) if search_times else float("nan"),
        "search_ms_ci":     1000 * _ci95(search_times) if search_times else float("nan"),
        "avg_scanned":      mean(scanned) if scanned else float("nan"),
        "avg_pool":         mean(pools) if pools else float("nan"),
    }
    if wrapper is not None:
        out["build_s"] = wrapper.get("preprocessing_time", float("nan"))
        out["index_MB"] = wrapper.get("index_memory_MB", float("nan"))
        out["raw_vectors_MB"] = wrapper.get("raw_vectors_MB", float("nan"))
    return out


def parse_results_dir(results_dir: str, queries: list[dict]) -> pd.DataFrame:
    """
    Walk results_dir, evaluate every pkl, return a DataFrame with one row per file.
    """
    rows = []
    for method in sorted(os.listdir(results_dir)):
        method_dir = os.path.join(results_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for fname in sorted(os.listdir(method_dir)):
            if not fname.endswith(".pkl"):
                continue
            fpath = os.path.join(method_dir, fname)
            params = fname.replace(".pkl", "")
            try:
                metrics = evaluate_file(fpath, queries)
            except Exception as e:
                import traceback
                print(f"[WARN] could not evaluate {fpath}:")
                traceback.print_exc()
                continue
            rows.append({
                "method":  method,
                "params":  params,
                **metrics,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["method", "params"]).reset_index(drop=True)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  required=True,
                        help="Top-level result folder, e.g. results_ranged_k=10_m=2_fdist=euclidean_200")
    parser.add_argument("--queries_path", required=True,
                        help="Path to the matching ranged queries pkl")
    parser.add_argument("--output",       default="summary.csv",
                        help="Where to write the CSV summary (default: summary.csv)")
    args = parser.parse_args()

    print(f"Loading queries from {args.queries_path} ...")
    with open(args.queries_path, "rb") as f:
        queries = pickle.load(f)
    print(f"  {len(queries)} queries loaded.")

    print(f"Evaluating results in {args.results_dir} ...")
    df = parse_results_dir(args.results_dir, queries)

    print("\n" + df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
