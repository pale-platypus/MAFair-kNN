"""
exact_cartesian.py
------------------
Exact per-partition retrieval: a vectorized brute-force scan over the
Cartesian-partitioned layout. For every combination of constrained values it
scans all points of the matching partitions, keeps the k_pi nearest (the
retrieval quota), and passes the pool to the same exact post-processing.
Because the pool contains the k_pi nearest points of every combination, the
result is the exact optimum of MAFair-KNN (it is the ground-truth procedure).
It is the "brute force" reference for latency comparisons.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import time
import itertools as itt
import numpy as np
from collections import defaultdict

from l2lsh_cartesian import L2LSHCartesian, reconstruct_metadata
from solver import build_solver


def _ub(c):
    return c[1] if isinstance(c, (tuple, list)) else c


class ExactCartesian(L2LSHCartesian):
    def __init__(self, args, d, metadata_store, protected_attrs, attrs_memmap=None, **kwargs):
        self.args = args
        self.d = d
        self.attrs_memmap = attrs_memmap
        self.metadata_store = metadata_store
        groups = defaultdict(list)
        for a in protected_attrs:
            kk, _ = a.split(":", 1)
            groups[kk].append(a)
        self.all_carte_attrs = ["__".join(sorted(t)) for t in itt.product(*groups.values())]
        self.partitions = self.group_ids_by_partition()
        self.partition_tokens = [(p, set(p.split("__"))) for p in self.partitions]
        self._build_token_index()
        self.blocks = {}

    def build_index(self, X, **kwargs):
        # partitioned layout: contiguous vectors + squared norms per partition
        for pi, ids in self.partitions.items():
            ids_arr = np.sort(np.asarray(ids, dtype=np.int64))
            V = np.ascontiguousarray(X[ids_arr], dtype=np.float32)
            self.blocks[pi] = (ids_arr, V, np.einsum("ij,ij->i", V, V))

    # ---- shared tail: ids -> metadata, de-duplicate, solve ----
    def _finish(self, pairs, query, search_time, total_scan):
        if self.attrs_memmap is not None:
            cands = [(reconstruct_metadata(int(i), self.attrs_memmap[int(i)]), float(d)) for i, d in pairs]
        else:
            cands = [(self.metadata_store[int(i)], float(d)) for i, d in pairs]
        seen, deduped = set(), []
        for meta, dist in sorted(cands, key=lambda x: x[1]):
            oid = meta.split("__")[0]
            if oid not in seen:
                seen.add(oid)
                deduped.append((meta, dist))
        if not deduped:
            return None
        t0 = time.perf_counter()
        res = build_solver(self.args).solve(deduped, query)
        res["postprocessing_time"] = time.perf_counter() - t0
        res["search_time"] = search_time
        res["total_scanned"] = total_scan
        res["pool_size"] = len(deduped)
        return res

    @staticmethod
    def _exact_topk(ids, vector_store, q, k):
        """Exact squared-L2 re-ranking (same arithmetic as the ground truth)."""
        V = np.asarray(vector_store[ids], dtype=np.float32)
        diff = V - q
        d = np.einsum("ij,ij->i", diff, diff)
        if len(d) > k:
            sel = np.argpartition(d, k - 1)[:k]
            return ids[sel], d[sel]
        return ids, d

    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
        data = query["count"]
        q = np.asarray(query.get("vector", query.get("text_query_embedding")), dtype=np.float32)
        qq = float(q @ q)
        lists = [[f"{a}:{v}" for v in vals] for a, vals in data.items()]
        pairs, total_scan = [], 0
        t0 = time.perf_counter()
        for combo in itt.product(*lists):
            k_pi = min(_ub(data[t.split(":", 1)[0]][t.split(":", 1)[1]]) for t in combo)
            parts = self._match(combo)
            if not parts or k_pi <= 0:
                continue
            margin = 2 * k_pi + 8           # fast-formula pre-selection, exact re-rank below
            ids_l = []
            for pi in parts:
                ids_arr, V, sq = self.blocks[pi]
                d = sq - 2.0 * (V @ q) + qq
                if len(d) > margin:
                    ids_l.append(ids_arr[np.argpartition(d, margin - 1)[:margin]])
                else:
                    ids_l.append(ids_arr)
                total_scan += len(ids_arr)
            ids_c = np.concatenate(ids_l)
            ids_c, d_c = self._exact_topk(ids_c, vector_store, q, k_pi)
            pairs.extend(zip(ids_c.tolist(), d_c.tolist()))
        return self._finish(pairs, query, time.perf_counter() - t0, total_scan)
