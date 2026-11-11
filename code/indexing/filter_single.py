"""
filter_single.py
----------------
Filtered search over ONE index of the whole dataset (no attribute partitioning),
the IVF and LSH counterparts of Filter-DiskANN: the search only admits points
whose value of every constrained attribute the query lists, and keeps the
K = F*k nearest admitted points it finds (F = --filtering-multiplier). The pool
then goes to the same exact post-processing (with the quota pruning of
global_postfilter, which does not change the optimum).

filter_ivf : FAISS IVF over all points (nlist = 4*sqrt(n)); the filter is applied
             while scanning the --nprobe closest lists, and nprobe is doubled until
             the probed lists hold K admitted points.
filter_lsh : --ell tables of --mu concatenated p-stable hashes of width --w over all
             points; every admitted point in the query's buckets is ranked exactly.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import time
import numpy as np
import faiss

from global_postfilter import GlobalPostFilter


class _FilteredSingle(GlobalPostFilter):
    def _index_attrs(self, n):
        super()._index_attrs(n)
        # one packed bitmap per attribute value, as a filtered index keeps them
        member = {}
        for pi, ids in self.partitions.items():
            ids = np.asarray(ids, dtype=np.int64)
            for tok in pi.split("__"):
                a, v = tok.split(":", 1)
                member.setdefault(a, {}).setdefault(v, np.zeros(n, dtype=bool))[ids] = True
        self.bitmaps = {a: {v: np.packbits(m, bitorder="little") for v, m in d.items()}
                        for a, d in member.items()}

    def _eligible(self, data):
        """packed bitmap over all points (bit i&7 of byte i>>3): admitted by the query"""
        bits = np.full((self.n + 7) // 8, 0xFF, dtype=np.uint8)
        for a, vals in data.items():
            acc = np.zeros_like(bits)
            for v in vals:
                bm = self.bitmaps[a].get(v)
                if bm is not None:
                    acc |= bm
            bits &= acc
        return bits

    @staticmethod
    def _bit(bits, ids):
        return ((bits[ids >> 3] >> (ids & 7).astype(np.uint8)) & 1).astype(bool)

    def _query(self, query):
        q = np.asarray(query.get("vector", query.get("text_query_embedding")), dtype=np.float32)
        return query["count"], q, int(self.args.filtering_multiplier) * int(query["k"])


class FilterIVF(_FilteredSingle):
    def build_index(self, X, **kwargs):
        self.args.global_type = "ivf"
        super().build_index(X, **kwargs)
        self.n = X.shape[0]

    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
        data, q, K = self._query(query)
        t0 = time.perf_counter()
        bits = self._eligible(data)
        sel = faiss.IDSelectorBitmap(self.n, faiss.swig_ptr(bits))
        # like a filtered graph search, keep searching until K admitted points are found:
        # double nprobe while the probed lists hold fewer than K admitted points
        nprobe = int(self.args.nprobe)
        while True:
            params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
            _, I = self.index.search(q.reshape(1, -1), K, params=params)
            ids = I[0][I[0] >= 0]
            if len(ids) >= K or nprobe >= self.index.nlist:
                break
            nprobe = min(2 * nprobe, self.index.nlist)
        pairs = self._quota_pool(ids, data, q, vector_store)
        return self._finish(pairs, query, time.perf_counter() - t0, len(ids))


class FilterLSH(_FilteredSingle):
    def build_index(self, X, **kwargs):
        n, d = X.shape
        self.n = n
        self._index_attrs(n)
        mu, ell, w = int(self.args.mu), int(self.args.ell), float(self.args.w)
        rng = np.random.default_rng(self.args.seed)
        self.A = rng.normal(size=(ell, mu, d)).astype(np.float32)
        self.b = rng.uniform(0.0, w, size=(ell, mu)).astype(np.float32)
        self.w = w
        # a table's key: its mu bucket indices combined by random odd 64-bit multipliers
        self.mix = rng.integers(1, 2**62, size=mu, dtype=np.int64).astype(np.uint64) | np.uint64(1)
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        self.keys = np.empty((ell, n), dtype=np.uint64)
        self.ids = np.empty((ell, n), dtype=np.int32)
        for t in range(ell):
            key = self._keys(Xf, t)
            order = np.argsort(key, kind="stable")
            self.keys[t], self.ids[t] = key[order], order

    def _keys(self, X, t):
        h = np.floor((X @ self.A[t].T + self.b[t]) / self.w).astype(np.int64)
        return (h.astype(np.uint64) * self.mix).sum(axis=1, dtype=np.uint64)

    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
        data, q, K = self._query(query)
        t0 = time.perf_counter()
        hit = np.zeros(self.n, dtype=bool)
        for t in range(self.keys.shape[0]):
            kq = self._keys(q[None, :], t)[0]
            lo = np.searchsorted(self.keys[t], kq, side="left")
            hi = np.searchsorted(self.keys[t], kq, side="right")
            hit[self.ids[t, lo:hi]] = True
        ids = np.flatnonzero(hit)
        scanned = len(ids)
        ids = ids[self._bit(self._eligible(data), ids)]
        if len(ids):
            ids, _ = self._exact_topk(ids, vector_store, q, K)
        pairs = self._quota_pool(ids, data, q, vector_store)
        return self._finish(pairs, query, time.perf_counter() - t0, scanned)
