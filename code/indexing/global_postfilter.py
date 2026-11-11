"""
global_postfilter.py
--------------------
Oversample-then-post-filter baseline: ONE index over the whole dataset (no
attribute partitioning). A query retrieves the K global nearest neighbours,
discards those whose attribute values the query does not admit, and passes the
rest to the same exact post-processing. Before solving, each value combination
keeps only its k_pi nearest retrieved points; by the retrieval-quota argument
this does not change the optimum over the retrieved set, it only shrinks the ILP.

--global-type flat : exact global top-K (an idealised, error-free global index)
--global-type ivf  : FAISS IVF over the whole dataset (nlist = 4*sqrt(n)), --nprobe
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import math
import time
import numpy as np
import faiss
faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

from exact_cartesian import ExactCartesian, _ub


class GlobalPostFilter(ExactCartesian):
    def build_index(self, X, **kwargs):
        n, d = X.shape
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        if self.args.global_type == "ivf":
            nlist = max(1, min(int(round(4 * math.sqrt(n))), n // 39))
            quant = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
            ntr = min(n, max(39 * nlist, 50_000))
            index.train(Xf[np.linspace(0, n - 1, ntr).astype(np.int64)])
            index.add(Xf)
            index.nprobe = max(1, int(self.args.nprobe))
        else:
            index = faiss.IndexFlatL2(d)
            index.add(Xf)
        self.index = index
        self._index_attrs(n)

    def _index_attrs(self, n):
        # attribute value of every point, per attribute (for post-filtering)
        self.attr_of = {}
        for pi, ids in self.partitions.items():
            ids = np.asarray(ids, dtype=np.int64)
            for tok in pi.split("__"):
                a, v = tok.split(":", 1)
                arr = self.attr_of.setdefault(a, np.empty(n, dtype=object))
                arr[ids] = v

    def _admitted(self, ids, data):
        """mask over ids: the point's value of every constrained attribute is admitted"""
        keep = np.ones(len(ids), dtype=bool)
        for a, vals in data.items():
            keep &= np.isin(self.attr_of[a][ids], list(vals.keys()))
        return keep

    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
        data = query["count"]
        q = np.asarray(query.get("vector", query.get("text_query_embedding")), dtype=np.float32)
        K = int(self.args.global_k)
        t0 = time.perf_counter()
        _, I = self.index.search(q.reshape(1, -1), K)
        ids = I[0][I[0] >= 0]
        ids = ids[self._admitted(ids, data)]
        pairs = self._quota_pool(ids, data, q, vector_store)
        return self._finish(pairs, query, time.perf_counter() - t0, K)

    def _quota_pool(self, ids, data, q, vector_store):
        # group by value combination, keep the k_pi nearest of each (quota pruning)
        pairs = []
        if len(ids):
            attrs = list(data.keys())
            keys = list(zip(*[self.attr_of[a][ids] for a in attrs]))
            groups = {}
            for i, key in zip(ids.tolist(), keys):
                groups.setdefault(key, []).append(i)
            for key, gids in groups.items():
                k_pi = min(_ub(data[a][v]) for a, v in zip(attrs, key))
                gids = np.asarray(gids, dtype=np.int64)
                gi, gd = self._exact_topk(gids, vector_store, q, k_pi)
                pairs.extend(zip(gi.tolist(), gd.tolist()))
        return pairs
