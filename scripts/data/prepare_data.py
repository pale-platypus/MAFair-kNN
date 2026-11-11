"""
prepare_data.py — build the standard-format (vectors.npz with 'vectors' and
'metadata') datasets used by the ICDE experiments.

  data/sift1m, data/sift10m : real SIFT/BIGANN vectors (first 1M / 10M of
      BIGANN) with the synthetic 3x4x3 attributes already drawn in
      data/bigann_{1m,10m}/attributes.dat (same schema as SIFT1B).
  data/sift1m/metadata_V{5,10,20,50}.npz : alternative attribute draws for the
      partition-scalability study: m = 3 attributes, |V_j| values each, drawn
      i.i.d. from a Zipf(s = 1) law over values, independently of the vectors.
  data/cbin : CelebA vectors with 10 real binary attributes.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = lambda *p: os.path.join(ROOT, "data", *p)


def meta_strings(codes, names, values):
    n = codes.shape[0]
    cols = [np.asarray(values[j], dtype=object)[codes[:, j]] for j in range(len(names))]
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = f"id:{i}__" + "__".join(f"{names[j]}:{cols[j][i]}" for j in range(len(names)))
    return out.astype(str)


def sift(src, dst):
    if os.path.exists(D(dst, "vectors.npz")):
        print("exists", dst); return
    info = np.load(D(src, "metadata.npz"), allow_pickle=True)
    n, d = int(info["n"][0]), int(info["d"][0])
    names = list(info["attr_names"]); values = [list(v) for v in info["attr_values"]]
    X = np.array(np.memmap(D(src, "vectors.dat"), dtype=np.float32, mode="r", shape=(n, d)))
    A = np.array(np.memmap(D(src, "attributes.dat"), dtype=np.uint8, mode="r", shape=(n, len(names))))
    os.makedirs(D(dst), exist_ok=True)
    np.savez(D(dst, "vectors.npz"), vectors=X, metadata=meta_strings(A, names, values))
    print("wrote", dst, X.shape)


def zipf_variants(dst, sizes=(5, 10, 20, 50), m=3, seed=7):
    n = int(np.load(D(dst, "vectors.npz"))["vectors"].shape[0])
    rng = np.random.default_rng(seed)
    for V in sizes:
        f = D(dst, f"metadata_V{V}.npz")
        if os.path.exists(f):
            print("exists", f); continue
        p = 1.0 / np.arange(1, V + 1); p /= p.sum()
        codes = np.stack([rng.choice(V, size=n, p=p) for _ in range(m)], axis=1)
        names = [f"A{j+1}" for j in range(m)]
        values = [[f"V{j+1}_{v+1}" for v in range(V)] for j in range(m)]
        np.savez(f, metadata=meta_strings(codes, names, values))
        print("wrote", f)


def celeba_binary(dst="cbin"):
    if os.path.exists(D(dst, "vectors.npz")):
        print("exists", dst); return
    X = np.load(D("celeba", "vectors.npz"))["vectors"]
    df = pd.read_csv(D("celeba", "list_attr_celeba.csv"))
    cols = ["Male", "Young", "Pale_Skin", "Heavy_Makeup", "Eyeglasses", "Smiling",
            "Black_Hair", "Bangs", "Mustache", "Wearing_Lipstick"]
    names = [c.replace("_", "") for c in cols]
    assert len(df) == len(X), (len(df), len(X))
    codes = np.stack([(df[c].values == 1).astype(np.int64) for c in cols], axis=1)
    values = [["no", "yes"]] * len(cols)
    os.makedirs(D(dst), exist_ok=True)
    np.savez(D(dst, "vectors.npz"), vectors=X, metadata=meta_strings(codes, names, values))
    print("wrote", dst, X.shape)


if __name__ == "__main__":
    what = sys.argv[1:] or ["sift1m", "variants", "cbin", "sift10m"]
    if "sift1m" in what: sift("bigann_1m", "sift1m")
    if "variants" in what: zipf_variants("sift1m")
    if "cbin" in what: celeba_binary()
    if "sift10m" in what: sift("bigann_10m", "sift10m")
