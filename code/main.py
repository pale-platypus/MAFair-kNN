import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# All methods run on ONE core (BLAS, OpenMP/faiss, hnswlib) so that build and
# query times are comparable across LSH (pure Python/numpy) and C++ libraries.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import tqdm
import math
import time
import json
import pickle
import pandas as pd
import numpy as np

sys.path.extend([
    '.',
    'code/indexing/',
])

from indexing import build_index
from arguments import get_args
from utils import (
    set_seed,
    get_dist_func,
)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    
    DATASET = args.data_dir.split("/")[-1]

    # ── billion-scale path: vectors.dat + attributes.dat memmaps ─────────────
    vec_dat  = os.path.join(args.data_dir, "vectors.dat")
    attr_dat = os.path.join(args.data_dir, "attributes.dat")
    is_billion = os.path.exists(vec_dat) and os.path.exists(attr_dat)

    if is_billion:
        meta_info   = np.load(os.path.join(args.data_dir, "metadata.npz"), allow_pickle=True)
        n_total     = int(meta_info["n"][0])
        d_total     = int(meta_info["d"][0])
        vector_store  = np.memmap(vec_dat,  dtype=np.float32, mode="r", shape=(n_total, d_total))
        attrs_memmap  = np.memmap(attr_dat, dtype=np.uint8,   mode="r", shape=(n_total, len(meta_info["attr_sizes"])))
        metadata_store = None   # not used; reconstructed on the fly
        # protected_attrs built from ATTR_NAMES / ATTR_VALUES in l2lsh_cartesian
        from l2lsh_cartesian import ATTR_NAMES, ATTR_VALUES
        protected_attrs = []
        for j, name in enumerate(ATTR_NAMES):
            for val in ATTR_VALUES[j]:
                protected_attrs.append(f"{name}:{val}")
        print(f"Billion-scale dataset: {n_total:,} vectors, d={d_total}, attrs={attrs_memmap.shape[1]}")
    else:
        # ── standard path ─────────────────────────────────────────────────────
        attrs_memmap = None
        npz_data = np.load(f"{args.data_dir}/vectors.npz")
        vector_store, metadata_store = npz_data['vectors'], npz_data['metadata']

        if args.metadata_file:
            metadata_store = np.load(args.metadata_file)['metadata']
        elif "celeb" in DATASET:
            metadata_store = np.load(f"{args.data_dir}/metadata_m={args.m}.npz")['metadata']

        if "paper" in DATASET:
            vector_store = vector_store[:1_000_000]
            metadata_store = metadata_store[:1_000_000]


    # """ uncomment this block for DEBUG
    # n_debug = 1000000
    # print(f"[Debug] Using first {n_debug:,} vectors")
    # vector_store  = vector_store[:n_debug]
    # if is_billion:
    #     attrs_memmap = attrs_memmap[:n_debug]
    #     n_total = n_debug
    # else:
    #     metadata_store = metadata_store[:n_debug]
    # """


    if args.fdist == "cosine":
        norms = np.linalg.norm(vector_store, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vector_store = vector_store / norms

    if not is_billion:
        metadata_store = [str(md) for md in metadata_store]
        protected_attrs = [md.split('__')[1:] for md in metadata_store]
        protected_attrs = list(set( [attr for sublist in protected_attrs for attr in sublist] ))

    """
    Compute R
    """
    if "lsh" in args.index and "ivf" not in args.index:
        # sample K points
        sample = vector_store[np.random.choice(len(vector_store), size=5000, replace=False)]
        nn_dists = []
        for i in range(len(sample)):
            d = np.linalg.norm(sample - sample[i], axis=1)
            d[i] = np.inf
            nn_dists.append(np.min(d))
        args.r = round(np.percentile(nn_dists, 50) * 2, 5)
        
    ## indexing params
    def make_index_name(args):
        if args.index == "filter_ivf":
            name = f"{args.index}/nprobe={args.nprobe}_mult={args.filtering_multiplier}"
        elif args.index == "filter_lsh":
            name = f"{args.index}/w={args.w}_ell={args.ell}_mu={args.mu}_mult={args.filtering_multiplier}"
        elif "ivf" in args.index:
            name = f"{args.index}/nlist={args.nlist}_nprobe={args.nprobe}_over={args.oversample}"
        elif "lsh" in args.index:
            name = f"{args.index}/c={args.c}_r={args.r:.3f}_w={args.w}_ell={args.ell}_mu={args.mu}_delta={args.delta}"
            if args.share_hash != 1 or args.scan_threshold:
                name += f"_sh={args.share_hash}_th={args.scan_threshold}"
        elif "global" in args.index:
            name = f"{args.index}/type={args.global_type}_K={args.global_k}_nprobe={args.nprobe}"
        elif "exact" in args.index:
            name = f"{args.index}/exact"
        elif "sieve" in args.index:
            name = f"{args.index}/efc={args.ef_construction}_M={args.M}_efs={args.ef_search}"
            if args.scan_threshold:
                name += f"_th={args.scan_threshold}"
        elif "disk" in args.index:
            name = f"{args.index}/efc={args.ef_construction}_M={args.M}_efs={args.ef_search}_mult={args.filtering_multiplier}"
        elif "brute" in args.index:
            name = f"{args.index}"
        else:
            raise ValueError(f"Index {args.index} is not configured to save outputs! Check main.py!!")
        return name + f"__solver={args.solver}"

    def set_query_param(db, args, param, value):
        """Change a QUERY-TIME parameter of a built index (no rebuild)."""
        setattr(args, param, value)
        if param == "nprobe":
            if hasattr(db, "nprobe"):
                db.nprobe = int(value)
            if hasattr(db, "index") and hasattr(db.index, "nprobe"):
                db.index.nprobe = int(value)
        elif param == "ef_search":
            if hasattr(db, "_subidx"):                    # SIEVE++: one HNSW per partition
                for sub in db._subidx.values():
                    if hasattr(sub, "ef_search"):
                        sub.ef_search = int(value)
                        sub.index.set_ef(int(value))
            if hasattr(db, "ef_search") and hasattr(db, "index"):   # Filter-DiskANN
                db.ef_search = int(value)
                db.index.set_ef(int(value))
        elif param in ("global_k", "filtering_multiplier"):
            pass                                          # read from args at query time
        else:
            raise ValueError(f"cannot sweep {param} without rebuilding")

    sweep_param, sweep_vals = None, [None]
    if getattr(args, "sweep", None):
        sweep_param, vals = args.sweep.split("=", 1)
        sweep_vals = [int(v) for v in vals.split(",")]
    INDEX = make_index_name(args)

    """
    indexing
    """
    import psutil, os as _os
    proc = psutil.Process(_os.getpid())

    start = time.perf_counter()
    index = build_index(args)
    db = index(
        args,
        vector_store.shape[-1],
        metadata_store,
        protected_attrs,
        attrs_memmap=attrs_memmap,
    )
    mem_before = proc.memory_info().rss
    db.build_index(
        vector_store,
        n_workers=4,
        chunk_size=500_000,
        use_cache=args.use_cache,
    )
    indexing_time = time.perf_counter() - start

    if is_billion:
        size_mb = (proc.memory_info().rss - mem_before) / (1024 * 1024)
        print(f"Memory cost (RSS delta): {size_mb:.1f} (MB)")
    else:
        data = pickle.dumps(db, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = len(data) / (1024 * 1024)
        print(f"Memory cost: {size_mb} (MB)")
    print(f"Preprocessing time: {indexing_time} (s)")
    
    """
    retrieval
    """
    dfunc = get_dist_func(args.fdist)

    avg_post = 0.
    # workloads to answer with this ONE index: --query-tag-list (query file with a {tag}
    # placeholder), an explicit --query-file, or the default naming over --k-list
    if getattr(args, "query_tag_list", None):
        workloads = [(args.query_file.format(tag=t), t) for t in args.query_tag_list.split(",")]
    elif getattr(args, "query_file", None):
        workloads = [(args.query_file, args.query_tag or os.path.splitext(os.path.basename(args.query_file))[0])]
    else:
        _klist = [int(x) for x in args.k_list.split(",")] if getattr(args, "k_list", None) else [args.k]
        workloads = [(f"{args.data_dir}/ranged_queries_k={k}_m={args.m}_fdist={args.fdist}_200.pkl",
                      f"k={k}_m={args.m}_fdist={args.fdist}_200") for k in _klist]
    for (query_path, query_suffix), sv in [(w, sv) for w in workloads for sv in sweep_vals]:
        if sweep_param is not None:
            set_query_param(db, args, sweep_param, sv)
            INDEX = make_index_name(args)
        with open(query_path, 'rb') as f:
            queries = pickle.load(f)

        result_path = f"{args.out_root}/{DATASET}/results_ranged_{query_suffix}/{INDEX}.pkl"
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        results = {}
        results["preprocessing_time"] = indexing_time
        results["index_memory_MB"] = size_mb
        results["raw_vectors_MB"] = float(vector_store.shape[0]) * vector_store.shape[1] * 4 / (1024 * 1024)
        results["args"] = dict(vars(args))

        results["query_results"] = []
        n_failed = 0
        for query in tqdm.tqdm(queries):
            result = db.search_and_solve(query, vector_store, dfunc)
            results["query_results"].append(result)
            if result is None:
                n_failed += 1
                continue
            avg_post += result["postprocessing_time"]

        if n_failed:
            print(f"[Warning] {n_failed}/{len(queries)} queries returned no candidates")

        with open(result_path, 'wb') as f:
            pickle.dump(results, f)

        # print(f"Post processing time for C_MIN={C_MIN}:")
        # print(avg_post/10)
        # print()