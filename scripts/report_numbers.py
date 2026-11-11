"""
numbers.py — print the defaults and the LaTeX table rows of the experimental section
straight from results/all_runs.csv (no manual transcription).
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full = pd.read_csv(os.path.join(ROOT, "results", "all_runs.csv"))
for c in ("mu", "ell", "nprobe", "efs", "mult", "K", "th"):
    if c in full:
        full[c] = pd.to_numeric(full[c], errors="coerce")
a = full[full.solver == "milp"]
# LSH- and IVF-baseline were run with --solver post_alg, which for m >= 3 is the same integer program
single = full[full.method.isin(["filter_lsh", "filter_ivf"]) & (full.solver == "post_alg")]


def default(ds, meth, knob, tag="k=10_m=3", target=0.95, extra=None):
    s = a[(a.dataset == ds) & (a.tag == tag) & (a.method == meth)]
    if extra is not None:
        s = s[extra(s)]
    if s.empty:
        return None
    ok = s[s["recall@k"] >= target]
    return ok.sort_values("avg_total_ms").iloc[0] if len(ok) else s.sort_values("recall@k").iloc[-1]


DS = ["audio", "fairface", "celeba", "sift1m", "sift10m"]
plain_lsh = lambda s: s["th"].isna() if "th" in s else np.ones(len(s), bool)
D = {}
print("== defaults (cheapest with recall >= 0.95 on k=10, m=3; else most accurate)")
for ds in DS:
    D[ds] = {
        "lsh": default(ds, "l2lsh_cartesian", ["mu", "ell"], extra=plain_lsh),
        "ivf": default(ds, "l2ivf_cartesian", ["nprobe"]),
        "sieve": default(ds, "sieve_cartesian", ["efs"]),
        "fda": default(ds, "filter_diskann", [], extra=lambda s: (s["mult"] == 4) & (s["efs"] == 32)),
        "exact": default(ds, "exact_cartesian", []),
        "global": default(ds, "global_postfilter", [], extra=lambda s: s["K"] == 1000),
    }
    s = single[(single.dataset == ds) & (single.tag == "k=10_m=3") & (single.method == "filter_ivf") & (single.mult == 4)]
    D[ds]["fivf"] = s.iloc[0] if len(s) else None
    for k, r in D[ds].items():
        if r is None:
            continue
        knob = {"lsh": f"mu={r.mu:.0f},ell={r.ell:.0f}", "ivf": f"nprobe={r.nprobe:.0f}", "sieve": f"EF={r.efs:.0f}"}.get(k, "")
        print(f"{ds:9s} {k:7s} {knob:18s} recall={r['recall@k']:.3f} succ={r['success_%']:.1f} "
              f"total={r.avg_total_ms:8.1f}ms search={r.avg_search_ms:8.1f} post={r.avg_post_ms:5.2f} "
              f"build={r.build_s:8.1f}s index={r.index_MB:9.1f}MB raw={r.raw_vectors_MB:7.1f}MB scanned={r.avg_scanned:.0f}")

print("\n== cost table rows: build (s) / index (GB)")
names = {"lsh": "Alg-Fair-LSH", "ivf": "Alg-Fair-IVF", "sieve": "Alg-Fair-HNSW", "exact": "Alg-Fair-Exact",
         "fivf": "IVF-baseline", "fda": "HNSW-baseline"}
cols = ["fairface", "celeba", "sift1m", "sift10m"]


def fmt_s(x):
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


for k, nm in names.items():
    cells = []
    for ds in cols:
        r = D[ds].get(k)
        c = "--" if r is None else f"{fmt_s(r.build_s)} / {r.index_MB / 1024:.2f}"
        if k == "lsh" and r is not None and r["recall@k"] < 0.5:
            c += r"$^\dagger$"
        cells.append(c)
    if k == "fivf":
        print(r"\midrule")
    print(nm + " & " + " & ".join(cells) + r" \\")
print("Raw vectors & " + " & ".join(f"-- / {D[ds]['exact'].raw_vectors_MB / 1024:.2f}" for ds in cols) + r" \\")

print("\n== partition scalability (SIFT1M, k=10, m=3)")
for tag, V in [("k=10_m=3", "3,4,3"), ("k=10_m=3_V5", "5"), ("k=10_m=3_V10", "10"), ("k=10_m=3_V20", "20"), ("k=10_m=3_V50", "50")]:
    g = a[(a.dataset == "sift1m") & (a.tag == tag)]
    ex = g[g.method == "exact_cartesian"]
    iv = g[(g.method == "l2ivf_cartesian") & (g.nprobe == 8)]
    ls = g[(g.method == "l2lsh_cartesian") & (g.mu == 12) & (g.ell == 64) & plain_lsh(g)]
    lt = g[(g.method == "l2lsh_cartesian") & (g.mu == 12) & (g.ell == 64) & (g["th"] == 1280)] if "th" in g else g.iloc[0:0]
    row = [V]
    for s, what in [(ex, ("avg_total_ms",)), (iv, ("recall@k", "avg_total_ms", "build_s", "index_MB")),
                    (ls, ("recall@k", "success_%", "avg_total_ms", "build_s", "index_MB")),
                    (lt, ("recall@k", "success_%", "avg_total_ms", "build_s", "index_MB"))]:
        row += [("--" if s.empty else f"{s.iloc[0][w]:.3g}") for w in what]
    print(" | ".join(row))

print("\n== single index vs. framework on FairFace and CelebA (k=10, m=3; abstract, Section I and Section V-D)")
print("candidates: K for the baselines, the pool of k_b nearest points per query bitmap for the framework")
for ds in ("fairface", "celeba"):
    g = full[(full.dataset == ds) & (full.tag == "k=10_m=3")]
    pick = lambda m, f: g[(g.method == m) & f(g)].iloc[0]
    rows = [("HNSW-baseline, K=2,560", pick("filter_diskann", lambda s: (s["mult"] == 256) & (s["efs"] == 32))),
            ("LSH-baseline, K=2,560", pick("filter_lsh", lambda s: s["mult"] == 256)),
            ("IVF-baseline, K=2,560", pick("filter_ivf", lambda s: s["mult"] == 256)),
            ("Global+PF, K=100,000", pick("global_postfilter", lambda s: s["K"] == 100000)),
            ("Alg-Fair-HNSW, EF=%d" % D[ds]["sieve"].efs, D[ds]["sieve"]),
            ("Alg-Fair-Exact", D[ds]["exact"])]
    pool = D[ds]["exact"].avg_pool
    print(f"-- {ds}")
    for nm, r in rows:
        cand = r.K if r.method == "global_postfilter" else r.mult * 10 if r.method.startswith("filter") else pool
        # hnswlib evaluates HNSW-baseline's filter in a Python callback at every visited node, so its time is not comparable
        t = "not timed" if r.method == "filter_diskann" else f"{r.avg_total_ms:7.1f} ms"
        print(f"   {nm:24s} candidates={cand:8,.0f} success={r['success_%']:5.1f}% recall={r['recall@k']:.3f} {t}")
    hn, ex, gl, iv = D[ds]["sieve"], D[ds]["exact"], rows[3][1], rows[2][1]
    print(f"   Alg-Fair-HNSW is {iv.avg_total_ms / hn.avg_total_ms:.1f}x faster than IVF-baseline and "
          f"{gl.avg_total_ms / hn.avg_total_ms:.1f}x faster than Global+PF; "
          f"Alg-Fair-Exact is {gl.avg_total_ms / ex.avg_total_ms:.1f}x faster than Global+PF")
