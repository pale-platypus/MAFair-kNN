"""
figures.py — the figures of the experimental section.

    python scripts/figures.py [quality] [postfilter] [pareto] [post] [billion]

Inputs : results/all_runs.csv (collect.py), results/billion_strict.csv, and for the
         post-processing figure the result pickles under outputs/ (report_post.py)
Outputs: results/figs/exp_<name>.pdf (+ .png previews)
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figs")
os.makedirs(OUT, exist_ok=True)

# ---- style: thin marks, recessive grid, fixed method -> colour/marker ----------
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.5,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# reference categorical palette, fixed order; colour follows the method everywhere
M = {
    "lsh":    dict(label="Alg-Fair-LSH",   color="#2a78d6", marker="o", ls="-"),
    "ivf":    dict(label="Alg-Fair-IVF",   color="#eb6834", marker="s", ls="-"),
    "sieve":  dict(label="Alg-Fair-HNSW",  color="#1baf7a", marker="^", ls="-"),
    "fda":    dict(label="HNSW-baseline",  color="#eda100", marker="v", ls="--"),
    "flsh":   dict(label="LSH-baseline",   color="#8c5a2b", marker="p", ls="--"),
    "fivf":   dict(label="IVF-baseline",   color="#1a9fb5", marker="h", ls="--"),
    "global": dict(label="Global+PF",       color="#e87ba4", marker="D", ls="--"),
    "exact":  dict(label="Alg-Fair-Exact", color="#0b0b0b", marker="*", ls="none"),
}
METHOD_DIR = {"l2lsh_cartesian": "lsh", "l2ivf_cartesian": "ivf", "sieve_cartesian": "sieve",
              "filter_diskann": "fda", "global_postfilter": "global",
              "exact_cartesian": "exact", "filter_lsh": "flsh", "filter_ivf": "fivf"}
COL1, COL2 = 3.45, 7.16          # IEEE column / page width (in)


def load():
    a = pd.read_csv(os.path.join(ROOT, "results", "all_runs.csv"))
    a["m_key"] = a["method"].map(METHOD_DIR)
    for c in ("mu", "ell", "nprobe", "efs", "mult", "K"):
        if c in a:
            a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a["solver"] == "milp") | a["solver"].isna()] if "solver" in a else a
    return a


def tag_k(tag):
    return int(re.search(r"k=(\d+)", tag).group(1))


def tag_m(tag):
    return int(re.search(r"m=(\d+)", tag).group(1))


DEFAULTS = {}     # filled by choose_defaults(): (dataset, m_key) -> row filter dict


def choose_defaults(a, target=0.95):
    """Per dataset, the cheapest setting reaching recall >= target on the default workload
    (k=10, m=3); if none reaches it, the most accurate setting."""
    for ds in sorted(a.dataset.unique()):
        d = a[(a.dataset == ds) & (a.tag == "k=10_m=3")]
        for mk, knob in [("lsh", ["mu", "ell"]), ("ivf", ["nprobe"]), ("sieve", ["efs"])]:
            s = d[d.m_key == mk]
            if s.empty:
                continue
            ok = s[s["recall@k"] >= target]
            best = ok.sort_values("avg_total_ms").iloc[0] if len(ok) else s.sort_values("recall@k").iloc[-1]
            DEFAULTS[(ds, mk)] = {c: best[c] for c in knob}
        DEFAULTS[(ds, "fda")] = {"mult": 4, "efs": 32}
        DEFAULTS[(ds, "flsh")] = {"mult": 4}
        DEFAULTS[(ds, "fivf")] = {"mult": 4}
        DEFAULTS[(ds, "global")] = {"K": 1000}
    return DEFAULTS


def pick(a, ds, tag, mk):
    s = a[(a.dataset == ds) & (a.tag == tag) & (a.m_key == mk)]
    f = DEFAULTS.get((ds, mk))
    if f:
        for c, v in f.items():
            s = s[s[c] == v]
    return s.iloc[0] if len(s) else None


# ---------------------------------------------------------------------------
def fig_quality(a):
    """recall@k (top) and success (bottom): FairFace vs k, Audio vs k, CelebA vs m."""
    cols = [("fairface", "FairFace", [f"k={k}_m=3" for k in (5, 10, 15, 20, 50)], tag_k, "$k$"),
            ("audio", "Audio", [f"k={k}_m=3" for k in (5, 10, 15, 20)], tag_k, "$k$"),
            ("celeba", "CelebA", [f"k=10_m={m}" for m in (1, 2, 3, 4, 5)], tag_m, "$m$")]
    methods = ["lsh", "ivf", "sieve", "fda", "flsh", "fivf", "global"]
    fig, ax = plt.subplots(2, 3, figsize=(COL2, 2.55), sharex="col")
    for j, (ds, name, tags, xf, xl) in enumerate(cols):
        for mk in methods:
            xs, rec, suc = [], [], []
            for t in tags:
                r = pick(a, ds, t, mk)
                if r is None:
                    continue
                xs.append(xf(t)); rec.append(r["recall@k"]); suc.append(r["success_%"])
            if not xs:
                continue
            st = M[mk]
            kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"] if st["ls"] != "none" else "-",
                      lw=1.2, ms=4, mfc="white" if st["marker"] not in ("x", "+") else st["color"], mew=1.0)
            ax[0, j].plot(xs, rec, **kw)
            ax[1, j].plot(xs, suc, **kw)
        ax[0, j].set_title(f"({'abc'[j]}) {name}", loc="left")
        ax[0, j].set_ylim(-0.04, 1.04)
        ax[1, j].set_ylim(-4, 104)
        ax[1, j].set_xlabel(xl)
        if ds != "celeba":
            ax[1, j].set_xscale("log")
            ticks = [tag_k(t) for t in tags]
            ax[1, j].set_xticks(ticks); ax[1, j].set_xticklabels([str(t) for t in ticks])
            ax[1, j].minorticks_off()
        else:
            ax[1, j].set_xticks([1, 2, 3, 4, 5])
    ax[0, 0].set_ylabel("Recall@$k$")
    ax[1, 0].set_ylabel("Successful (%)")
    handles = [Line2D([], [], color=M[k]["color"], marker=M[k]["marker"], ls=M[k]["ls"], lw=1.2, ms=4,
                      mfc="white" if M[k]["marker"] not in ("x", "+") else M[k]["color"], label=M[k]["label"])
               for k in methods]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.06),
               handlelength=2.0, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    save(fig, "quality")


def fig_postfilter(a, depth):
    """success vs K for the single indexes: exact global top-K + post-filtering, and filtered
    search keeping K = F k admitted points with a graph, LSH or IVF index."""
    wl = [("fairface", "k=10_m=3", "FairFace ($m$=3)"), ("celeba", "k=10_m=1", "CelebA ($m$=1)"),
          ("celeba", "k=10_m=3", "CelebA ($m$=3)"), ("celeba", "k=10_m=5", "CelebA ($m$=5)"),
          ("audio", "k=10_m=3", "Audio ($m$=3)")]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    mks = ["o", "s", "^", "v", "D"]
    panels = [("global", "K", 1, "(a) Global+PF"), ("fda", "mult", 10, "(b) HNSW-baseline"),
              ("flsh", "mult", 10, "(c) LSH-baseline"), ("fivf", "mult", 10, "(d) IVF-baseline")]
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 2.55), sharey=True, sharex=True)
    ax = axs.ravel()
    for (ds, tag, lab), c, mk in zip(wl, cols, mks):
        for i, (key, knob, scale, _) in enumerate(panels):
            g = a[(a.dataset == ds) & (a.tag == tag) & (a.m_key == key)]
            if key == "fda":
                g = g[g["efs"] == 32]
            g = g.sort_values(knob)
            if len(g):
                ax[i].plot(g[knob] * scale, g["success_%"], color=c, marker=mk, ms=3.5, lw=1.1, mfc="white", label=lab)
    for i, (_, _, _, t) in enumerate(panels):
        ax[i].set_xscale("log"); ax[i].set_title(t, loc="left")
        ax[i].set_xlim(6, 1.6e5); ax[i].set_xticks([10, 1e2, 1e3, 1e4, 1e5]); ax[i].minorticks_off()
        if i >= 2:
            ax[i].set_xlabel("Retrieved $K$")
        if i % 2 == 0:
            ax[i].set_ylabel("Successful (%)")
    ax[0].set_ylim(-4, 104)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.1), columnspacing=0.8,
               handlelength=1.6)
    fig.tight_layout(w_pad=0.5, h_pad=0.4)
    save(fig, "postfilter")


def frontier(xs, ys):
    """Pareto frontier (min x, max y) of a point cloud, sorted by x."""
    pts = sorted(zip(xs, ys))
    out, best = [], -1
    for x, y in pts:
        if y > best:
            out.append((x, y)); best = y
    return zip(*out) if out else ([], [])


def fig_pareto(a):
    panels = [("fairface", "k=10_m=3", "FairFace"), ("celeba", "k=10_m=3", "CelebA ($m$=3)"),
              ("sift1m", "k=10_m=3", "SIFT1M"), ("sift10m", "k=10_m=3", "SIFT10M")]
    panels = [p for p in panels if len(a[(a.dataset == p[0]) & (a.tag == p[1])])]
    fig, ax = plt.subplots(1, len(panels), figsize=(COL2, 1.55), squeeze=False)
    ax = ax[0]
    for i, (ds, tag, name) in enumerate(panels):
        d = a[(a.dataset == ds) & (a.tag == tag)]
        for mk, knob in [("lsh", None), ("ivf", "nprobe"), ("sieve", "efs"), ("global", "K"),
                         ("flsh", "mult"), ("fivf", "mult")]:
            s = d[d.m_key == mk]
            if s.empty:
                continue
            st = M[mk]
            if mk == "lsh":
                ax[i].scatter(s["avg_total_ms"], s["recall@k"], s=6, color=st["color"], alpha=0.35, lw=0)
                fx, fy = frontier(s["avg_total_ms"], s["recall@k"])
                ax[i].plot(list(fx), list(fy), color=st["color"], marker=st["marker"], ms=3.2, lw=1.1, mfc="white")
            else:
                s = s.sort_values(knob)
                ax[i].plot(s["avg_total_ms"], s["recall@k"], color=st["color"], marker=st["marker"], ms=3.2,
                           lw=1.1, ls=st["ls"], mfc="white")
        e = d[d.m_key == "exact"]
        if len(e):
            ax[i].plot(e["avg_total_ms"], e["recall@k"], marker="*", color="#0b0b0b", ms=7, ls="none")
        ax[i].set_xscale("log"); ax[i].set_title(f"({'abcd'[i]}) {name}", loc="left")
        ax[i].set_xlabel("Query time (ms)"); ax[i].set_ylim(-0.03, 1.03)
    ax[0].set_ylabel("Recall@10")
    keys = ["lsh", "ivf", "sieve", "exact", "flsh", "fivf", "global"]
    handles = [Line2D([], [], color=M[k]["color"], marker=M[k]["marker"], ls=M[k]["ls"] if k != "exact" else "none",
                      ms=4 if k != "exact" else 7, lw=1.1, mfc="white" if k != "exact" else M[k]["color"],
                      label=M[k]["label"]) for k in keys]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "pareto")


def greedy_vs_exact(ds, tag):
    """(% of queries greedy fails on, % it answers sub-optimally) on the exact pools of ds/tag,
    among the queries whose pool holds a fair answer (the integer program succeeds)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from report_post import compare
    from check_post import load as load_pkl, queries
    c = compare(load_pkl(ds, tag, "greedy"), load_pkl(ds, tag, "milp"), queries(ds, tag))
    return c["fail_pct"], c["fail_pct"] + c["costlier_pct"]


def fig_post(a_all):
    """post-processing on exact pools. Time: (a) vs k at m = 1 (Post-Alg-1 vs the ILP), (b) vs k at
    m = 2 (Post-Alg-2 vs the ILP), both on FairFace; (c) the ILP vs m on CelebA-B (k = 10).
    (d) greedy selection from the same pools vs m: queries it fails on / answers sub-optimally."""
    d = a_all[(a_all.dataset == "fairface") & (a_all.m_key == "exact")].copy()
    d["k"] = d.tag.map(tag_k)
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 2.75))
    ax = axs.ravel()
    alg = {1: ("post_alg1", "Post-Alg-1", "#6250d6", "D"), 2: ("post_alg2", "Post-Alg-2 (min-cost flow)", "#2a78d6", "o")}
    ilp = [("milp", "ILP (HiGHS)", "#eb6834", "s"), ("ilp", "ILP (CBC)", "#1baf7a", "^")]
    for i, m in enumerate((1, 2)):
        dm = d[d.tag.str.endswith(f"_m={m}")]
        for s, lab, c, mk in [alg[m]] + ilp:
            g = dm[dm.solver == s].sort_values("k")
            ax[i].plot(g["k"], g["avg_post_ms"], color=c, marker=mk, ms=3, lw=1.0, mfc="white", label=lab)
        ax[i].set_xscale("log"); ax[i].set_xlabel("$k$")
        ax[i].set_xticks([10, 100, 1000]); ax[i].minorticks_off()
        ax[i].set_title(f"({'ab'[i]}) FairFace, $m$={m}", loc="left")
    c = a_all[(a_all.dataset == "cbin") & (a_all.m_key == "exact")].copy()
    c["m"] = c.tag.map(tag_m)
    for s, lab, col, mk in ilp:
        g = c[c.solver == s].sort_values("m")
        ax[2].plot(g["m"], g["avg_post_ms"], color=col, marker=mk, ms=3, lw=1.0, mfc="white", label=lab)
    ax[2].set_xlabel("$m$"); ax[2].set_title("(c) CelebA-B, $k$=10", loc="left")
    ax[2].set_xticks([2, 4, 6, 8, 10])
    for a_ in ax[:3]:
        a_.set_yscale("log"); a_.set_ylim(0.03, 500)
    ax[0].set_ylabel("Post-processing (ms)"); ax[2].set_ylabel("Post-processing (ms)")
    for ds, lab, col, mk, ms_ in [("celeba", "CelebA", "#eda100", "v", [1, 2, 3, 4, 5]),
                                  ("cbin", "CelebA-B", "#e87ba4", "D", [2, 3, 4, 5, 6, 8, 10])]:
        v = [greedy_vs_exact(ds, f"k=10_m={m}")[0] for m in ms_]
        ax[3].plot(ms_, v, color=col, marker=mk, ms=3, lw=1.0, mfc="white", label=lab)
    ax[3].set_xlabel("$m$"); ax[3].set_ylabel("No fair answer (%)"); ax[3].set_ylim(-3, 80)
    ax[3].set_xticks([1, 2, 4, 6, 8, 10]); ax[3].set_title("(d) Greedy, $k$=10", loc="left")
    ax[3].legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.6, borderaxespad=0.3)
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend([h0[0], h1[0]] + h0[1:], [l0[0], l1[0]] + l0[1:], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.09), columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=0.4, h_pad=0.3)
    save(fig, "post")


def fig_billion():
    d = pd.read_csv(os.path.join(ROOT, "results", "billion_strict.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(COL1, 2.35), sharex=True)
    ser = [("SIFT1B", "#2a78d6", "o"), ("Synthetic 2", "#eb6834", "s")]
    panels = [("build_h", "Build time (h)", False), ("index_GB", "Index size (GB)", False),
              ("search_s", "Query time (s)", True), ("success", "Successful (%)", False)]
    panels[1] = ("index_GB", "Index size (GB)", True)
    for i, (col, lab, logy) in enumerate(panels):
        a = ax[i // 2, i % 2]
        for name, c, mk in ser:
            g = d[d.dataset == name].sort_values("ell")
            a.plot(g["ell"], g[col], color=c, marker=mk, ms=3.5, lw=1.1, mfc="white", label=name)
        a.set_xscale("log", base=2); a.set_xticks([4, 8, 16, 32, 64]); a.set_xticklabels(["4", "8", "16", "32", "64"])
        a.minorticks_off(); a.set_ylabel(lab); a.set_title(f"({'abcd'[i]})", loc="left")
        if logy:
            a.set_yscale("log")
        if i >= 2:
            a.set_xlabel(r"$\ell$")
    ax[1, 1].set_ylim(-4, 104)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=0.3, w_pad=0.6)
    save(fig, "billion")


def save(fig, name):
    fig.savefig(os.path.join(OUT, "exp_" + name + ".pdf"))
    fig.savefig(os.path.join(OUT, "exp_" + name + ".png"), dpi=200)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
# tables (LaTeX row fragments, written to results/tables.tex)
def dataset_stats():
    rows = []
    for ds, metaf in [("audio", None), ("fairface", None), ("celeba", "data/celeba/metadata_m=5.npz"),
                      ("cbin", None), ("sift1m", None), ("sift10m", None)]:
        p = os.path.join(ROOT, metaf) if metaf else os.path.join(ROOT, "data", ds, "vectors.npz")
        if not os.path.exists(p):
            continue
        M = np.load(p, allow_pickle=True)["metadata"]
        toks = [str(s).split("__")[1:] for s in M[: min(len(M), 2_000_000)]]
        attrs = {}
        for t in toks:
            for kv in t:
                a, v = kv.split(":", 1)
                attrs.setdefault(a, set()).add(v)
        cells = int(np.prod([len(v) for v in attrs.values()]))
        nonempty = len(set("__".join(t) for t in toks))
        rows.append(dict(dataset=ds, n=len(M), attrs={a: len(v) for a, v in attrs.items()}, cells=cells,
                         nonempty=nonempty))
        print(rows[-1])
    return rows


def acceptance():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "*", "icde", "*_stats.json"))):
        d = json.load(open(f))
        tag = os.path.basename(f).replace("ranged_queries_", "").replace("_fdist=euclidean_200_stats.json", "")
        rows.append(dict(dataset=d["dataset"], tag=tag, drawn=d["n_drawn"], valid=d["n_valid"],
                         frac=d["feasible_fraction_of_drawn"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a_all = pd.read_csv(os.path.join(ROOT, "results", "all_runs.csv"))
    a_all["m_key"] = a_all["method"].map(METHOD_DIR)
    for c in ("mu", "ell", "nprobe", "efs", "mult", "K"):
        if c in a_all:
            a_all[c] = pd.to_numeric(a_all[c], errors="coerce")
    # the framework's post-processing (solver=post_alg) where it was run, else the integer
    # program (solver=milp), which is what post_alg runs for m >= 3
    a = a_all[a_all.solver.isin(["milp", "post_alg"])].copy()
    a["base"] = a["params"].str.split("__solver=").str[0]
    a = a.sort_values("solver", key=lambda s: s.eq("milp")).drop_duplicates(
        ["dataset", "tag", "method", "base"], keep="first")
    choose_defaults(a)
    print({k: v for k, v in DEFAULTS.items()})
    which = sys.argv[1:] or ["quality", "postfilter", "pareto", "post", "billion"]
    if "quality" in which:
        fig_quality(a)
    if "postfilter" in which:
        fig_postfilter(a, None)
    if "pareto" in which:
        fig_pareto(a)
    if "post" in which:
        fig_post(a_all)
    if "billion" in which:
        fig_billion()
