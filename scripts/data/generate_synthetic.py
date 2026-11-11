"""
generate_synthetic.py
-------------------
Generates a synthetic vector dataset of N=1,000,000,000 vectors with 3
protected attributes, written to disk in chunks via numpy memmap.

Output files (in --out_dir):
  vectors.dat       float32 memmap, shape (N, d)
  attributes.dat    uint8   memmap, shape (N, 3)   — integer codes
  metadata.npz      lookup tables: attr names, value names, proportions
  vectors.npz       small verification slice (first 10k rows)

Attribute schema (mirrors the paper's notation):
  A1: 3 values  V11, V12, V13          (e.g. gender)
  A2: 4 values  V21, V22, V23, V24     (e.g. race/ethnicity)
  A3: 3 values  V31, V32, V33          (e.g. age group)
  → 36 Cartesian partitions total

Each partition π = (v1, v2, v3) gets its own Gaussian cluster center
drawn once at the start; points within the partition are sampled as
  x ~ N(center_π, σ²I_d)
This gives the "many clusters" structure that makes fair retrieval hard.

Memory budget: processes CHUNK rows at a time (~400 MB for d=128).
Disk budget:
  vectors.dat  ~512 GB  (1e9 × 128 × 4 bytes)
  attributes   ~3 GB    (1e9 × 3 × 1 byte)

Usage
-----
python scripts/data/generate_synthetic.py \\
    --n 1_000_000_000 \\
    --d 128 \\
    --out_dir /path/to/output \\
    --chunk 1_000_000 \\
    --seed 42

For a quick smoke-test use --n 1_000_000.
"""

import os
import time
import argparse
import numpy as np
from numpy.random import default_rng


# ── attribute schema ──────────────────────────────────────────────────────────

ATTR_NAMES = ["A1", "A2", "A3"]
ATTR_VALUES = [
    ["V11", "V12", "V13"],           # A1: 3 values
    ["V21", "V22", "V23", "V24"],    # A2: 4 values
    ["V31", "V32", "V33"],           # A3: 3 values
]
ATTR_SIZES = [len(v) for v in ATTR_VALUES]   # [3, 4, 3]
N_PARTITIONS = 1
for s in ATTR_SIZES:
    N_PARTITIONS *= s                         # 36

# Proportion of each attribute value (sums to 1 per attribute).
# Slightly imbalanced to mimic real datasets.
ATTR_PROPS = [
    np.array([0.40, 0.35, 0.25]),                    # A1
    np.array([0.30, 0.25, 0.25, 0.20]),              # A2
    np.array([0.40, 0.35, 0.25]),                    # A3
]
# Cartesian product proportion for each partition
def partition_proportions():
    """Return (36,) array of partition proportions."""
    import itertools
    props = []
    for idxs in itertools.product(*[range(s) for s in ATTR_SIZES]):
        p = 1.0
        for j, i in enumerate(idxs):
            p *= ATTR_PROPS[j][i]
        props.append(p)
    props = np.array(props)
    props /= props.sum()   # normalise numerical errors
    return props

# ── metadata string builder ───────────────────────────────────────────────────

def build_metadata_str(obj_id: int, a1: int, a2: int, a3: int) -> str:
    """Convert integer attribute codes to the paper's metadata string format.
    e.g. 'id:0__A1:V12__A2:V23__A3:V31'
    """
    return (
        f"id:{obj_id}"
        f"__A1:{ATTR_VALUES[0][a1]}"
        f"__A2:{ATTR_VALUES[1][a2]}"
        f"__A3:{ATTR_VALUES[2][a3]}"
    )

# ── main generation ───────────────────────────────────────────────────────────

def generate(n: int, d: int, out_dir: str, chunk: int, seed: int,
             sigma: float, center_scale: float):
    os.makedirs(out_dir, exist_ok=True)
    rng = default_rng(seed)

    import itertools

    # ── 1. Pre-compute cluster centers for all 36 partitions ──────────────────
    print(f"Sampling {N_PARTITIONS} cluster centers (d={d}) ...")
    centers = {}   # (a1_idx, a2_idx, a3_idx) → np.ndarray(d,)
    for idxs in itertools.product(*[range(s) for s in ATTR_SIZES]):
        centers[idxs] = rng.normal(0, center_scale, size=d).astype(np.float32)

    # ── 2. Pre-compute how many points each partition gets ────────────────────
    props = partition_proportions()
    part_list = list(itertools.product(*[range(s) for s in ATTR_SIZES]))
    counts_float = props * n
    counts = np.floor(counts_float).astype(np.int64)
    remainder = n - counts.sum()
    # distribute remainder to partitions with largest fractional part
    fracs = counts_float - counts
    top_idx = np.argsort(-fracs)[:remainder]
    counts[top_idx] += 1
    assert counts.sum() == n, "count mismatch"

    print("Partition counts (first 5):", counts[:5].tolist(), "...")
    print(f"Total vectors: {counts.sum():,}")

    # ── 3. Allocate memmap files ──────────────────────────────────────────────
    vec_path  = os.path.join(out_dir, "vectors.dat")
    attr_path = os.path.join(out_dir, "attributes.dat")

    print(f"Allocating vectors.dat  ({n * d * 4 / 1e9:.1f} GB) ...")
    vectors = np.memmap(vec_path,  dtype=np.float32, mode="w+", shape=(n, d))
    print(f"Allocating attributes.dat ({n * 3 / 1e9:.1f} GB) ...")
    attrs   = np.memmap(attr_path, dtype=np.uint8,   mode="w+", shape=(n, 3))

    # ── 4. Fill in partition by partition, chunk by chunk ─────────────────────
    write_ptr = 0
    t0 = time.time()

    for pi, idxs in enumerate(part_list):
        cnt = int(counts[pi])
        if cnt == 0:
            continue
        center = centers[idxs]
        a_codes = np.array(idxs, dtype=np.uint8)  # shape (3,)

        filled = 0
        while filled < cnt:
            batch = min(chunk, cnt - filled)
            noise = rng.normal(0, sigma, size=(batch, d)).astype(np.float32)
            vectors[write_ptr : write_ptr + batch] = center + noise
            attrs  [write_ptr : write_ptr + batch] = a_codes   # broadcast
            write_ptr += batch
            filled    += batch

        if (pi + 1) % 9 == 0 or pi == N_PARTITIONS - 1:
            elapsed = time.time() - t0
            pct = 100.0 * write_ptr / n
            rate = write_ptr / elapsed / 1e6
            print(f"  partition {pi+1:2d}/{N_PARTITIONS}  "
                  f"written {write_ptr:,} ({pct:.1f}%)  "
                  f"{rate:.1f}M vec/s  "
                  f"elapsed {elapsed:.0f}s")

    print("Flushing vectors.dat ...")
    vectors.flush()
    print("Flushing attributes.dat ...")
    attrs.flush()

    # ── 5. Save metadata lookup + small verification slice ───────────────────
    meta_path = os.path.join(out_dir, "metadata.npz")
    np.savez(
        meta_path,
        attr_names  = np.array(ATTR_NAMES),
        attr_values = np.array(ATTR_VALUES, dtype=object),
        attr_sizes  = np.array(ATTR_SIZES),
        attr_props  = np.array([p.tolist() for p in ATTR_PROPS], dtype=object),
        partition_props = props,
        n = np.array([n]),
        d = np.array([d]),
        sigma = np.array([sigma]),
    )
    print(f"Saved metadata to {meta_path}")

    # Small verification slice in the paper's .npz format
    verify_n = min(10_000, n)
    vecs_small = np.array(vectors[:verify_n])   # force load from memmap
    # build metadata strings for the slice
    meta_strings = np.array([
        build_metadata_str(i, int(attrs[i, 0]), int(attrs[i, 1]), int(attrs[i, 2]))
        for i in range(verify_n)
    ])
    verify_path = os.path.join(out_dir, "vectors.npz")
    np.savez(verify_path, vectors=vecs_small, metadata=meta_strings)
    print(f"Saved verification slice ({verify_n} rows) to {verify_path}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n── Generation complete ──────────────────────────────────────")
    print(f"  Vectors : {n:,}  ×  {d}d  float32")
    print(f"  Disk    : vectors.dat  {n*d*4/1e9:.1f} GB")
    print(f"            attributes.dat  {n*3/1e9:.1f} GB")
    print(f"  Time    : {total_time:.0f}s  ({n/total_time/1e6:.1f}M vec/s)")
    print(f"  Output  : {out_dir}")
    print()
    print("Sample metadata strings:")
    for i in range(min(5, verify_n)):
        print(f"  {meta_strings[i]}")


# ── build_metadata_store: reconstruct full metadata array from memmap ─────────

def build_metadata_store(out_dir: str, chunk: int = 1_000_000):
    """
    Reads vectors.dat + attributes.dat and writes a full metadata.dat
    (object array of strings) in chunks.  Only needed if downstream code
    requires the string-format metadata_store rather than the integer codes.

    Writes: metadata_strings.npy  (numpy object array, shape (n,))

    This is optional and slow (~minutes for 1B rows).  Most pipelines can
    reconstruct the string on-the-fly from the integer attributes.dat.
    """
    meta_path = os.path.join(out_dir, "metadata.npz")
    info = np.load(meta_path, allow_pickle=True)
    n = int(info["n"][0])

    attr_path = os.path.join(out_dir, "attributes.dat")
    attrs = np.memmap(attr_path, dtype=np.uint8, mode="r", shape=(n, 3))

    out_path = os.path.join(out_dir, "metadata_strings.npy")
    # Pre-allocate object array on disk via a regular numpy save is tricky;
    # build in batches and concatenate.
    print(f"Building metadata string array for {n:,} rows ...")
    all_strings = []
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch = attrs[start:end]
        strs = [
            build_metadata_str(start + i, int(batch[i, 0]), int(batch[i, 1]), int(batch[i, 2]))
            for i in range(end - start)
        ]
        all_strings.extend(strs)
        if (start // chunk) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {end:,}/{n:,}  ({100.*end/n:.1f}%)  {elapsed:.0f}s")
    arr = np.array(all_strings, dtype=object)
    np.save(out_path, arr)
    print(f"Saved {out_path}  ({arr.nbytes / 1e9:.2f} GB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate billion-scale synthetic dataset")
    p.add_argument("--n",            type=int,   default=1_000_000_000,
                   help="Number of vectors (default 1B; use 1_000_000 to smoke-test)")
    p.add_argument("--d",            type=int,   default=128,
                   help="Embedding dimension (default 128)")
    p.add_argument("--out_dir",      type=str,   required=True,
                   help="Output directory")
    p.add_argument("--chunk",        type=int,   default=1_000_000,
                   help="Rows per write batch (default 1M, ~400MB for d=128)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--sigma",        type=float, default=1.0,
                   help="Intra-cluster std dev (default 1.0)")
    p.add_argument("--center_scale", type=float, default=10.0,
                   help="Std dev of cluster center positions (default 10.0)")
    p.add_argument("--build_strings", action="store_true",
                   help="Also build the full metadata string array (slow, optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        n            = args.n,
        d            = args.d,
        out_dir      = args.out_dir,
        chunk        = args.chunk,
        seed         = args.seed,
        sigma        = args.sigma,
        center_scale = args.center_scale,
    )
    if args.build_strings:
        build_metadata_store(args.out_dir, chunk=args.chunk)
