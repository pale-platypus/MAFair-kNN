"""
collect.py — evaluate every result pickle under an output root and write one
CSV row per run.

    python scripts/collect.py [--root outputs] [--out results/all_runs.csv]

Result layout : <root>/<dataset>/results_ranged_<tag>/<method>/<params>.pkl
Query file    : data/<dataset>/icde/ranged_queries_<tag>_fdist=euclidean_200.pkl
"""
import argparse, os, pickle, re, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate_ranged import evaluate_file  # noqa: E402

_cache = {}


def queries_for(ds, tag):
    p = os.path.join(ROOT, "data", ds, "icde", f"ranged_queries_{tag}_fdist=euclidean_200.pkl")
    if p not in _cache:
        _cache[p] = pickle.load(open(p, "rb")) if os.path.exists(p) else None
    return _cache[p]


def parse_params(s):
    out = {}
    for part in re.split(r"_(?=[A-Za-z]+=)", s.replace("__", "_")):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", default="results/all_runs.csv")
    ap.add_argument("--filter", default="", help="only datasets/tags containing this substring")
    a = ap.parse_args()
    rows = []
    root = os.path.join(ROOT, a.root)
    for ds in sorted(os.listdir(root)):
        dsd = os.path.join(root, ds)
        if not os.path.isdir(dsd):
            continue
        for rd in sorted(os.listdir(dsd)):
            if not rd.startswith("results_ranged_"):
                continue
            tag = rd[len("results_ranged_"):]
            if a.filter and a.filter not in f"{ds}/{tag}":
                continue
            Q = queries_for(ds, tag)
            if Q is None:
                print("no query file for", ds, tag); continue
            for meth in sorted(os.listdir(os.path.join(dsd, rd))):
                md = os.path.join(dsd, rd, meth)
                if not os.path.isdir(md):
                    continue
                for f in sorted(os.listdir(md)):
                    if not f.endswith(".pkl"):
                        continue
                    params = f[:-4]
                    try:
                        m = evaluate_file(os.path.join(md, f), Q)
                    except Exception as e:  # keep going
                        print("FAILED", ds, tag, meth, params, e); continue
                    row = {"dataset": ds, "tag": tag, "method": meth, "params": params}
                    row.update(parse_params(params))
                    row.update(m)
                    rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    df.to_csv(os.path.join(ROOT, a.out), index=False)
    print(f"{len(df)} runs -> {a.out}")


if __name__ == "__main__":
    main()
