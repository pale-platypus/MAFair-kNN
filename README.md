# MAFair-KNN

Code, workloads and scripts to reproduce the experiments of the paper on fair *k*-nearest-neighbor retrieval under multi-attribute count constraints (MAFair-KNN, ICDE 2027 submission).

A query asks for the *k* points nearest to a query vector such that, for each of *m* protected attributes, the number of returned points with each value lies within given bounds. The framework partitions the data by attribute-value combination, retrieves a bounded number of near points from each partition relevant to the query, and post-processes the pool exactly: Post-Alg-1 for *m* = 1, Post-Alg-2 (min-cost flow) for *m* = 2, and an integer program for *m* ≥ 3.

## Methods

| Paper name | `--index` | Notes |
|---|---|---|
| Alg-Fair-LSH | `l2lsh_cartesian` | *ℓ* tables of *μ* p-stable hashes per partition |
| Alg-Fair-IVF | `l2ivf_cartesian` | FAISS `IndexIVFFlat` per partition |
| Alg-Fair-HNSW | `sieve_cartesian` | hnswlib graph per partition |
| Alg-Fair-Exact | `exact_cartesian` | no index: scans the relevant partitions |
| HNSW-baseline | `filter_diskann` | one HNSW graph over all points, filtered search |
| LSH-baseline | `filter_lsh` | one LSH index over all points, filtered search |
| IVF-baseline | `filter_ivf` | one IVF index over all points, filtered search |
| Global+PF | `global_postfilter` | exact global top-*K*, then post-filtering |

| Post-processing | `--solver` |
|---|---|
| by *m* (Post-Alg-1 / Post-Alg-2 / ILP) | `post_alg` |
| Post-Alg-1 | `post_alg1` |
| Post-Alg-2 (OR-Tools min-cost flow) | `post_alg2` |
| integer program, HiGHS | `milp` |
| integer program, CBC | `ilp` |
| greedy selection | `greedy` |

## Setup

Python 3.11:

```bash
pip install -r requirements.txt
```

All commands run from the repository root. Every method runs on one CPU thread (set in `code/main.py`). The paper's numbers were measured on an Intel Core i9-11900K with 128 GB of RAM, with at most four runs at a time. Query times therefore depend on the machine and its load, but success rates, recall, pool sizes and points scanned do not.

## Data

Every dataset lives in `data/<name>/`. Its `vectors.npz` holds two arrays:
- `vectors`: float32, shape (n, d).
- `metadata`: one string per point, of the form `id:<i>__<attr>:<value>__<attr>:<value>...`.

| Dataset | `data/` folder | How to obtain |
|---|---|---|
| Audio | `audio/vectors.npz` | our preprocessed file: **TODO: link** |
| FairFace | `fairface/vectors.npz` | our preprocessed file: **TODO: link** |
| CelebA | `celeba/vectors.npz`, `celeba/metadata_m={1..5}.npz`, `celeba/list_attr_celeba.csv` | our preprocessed files: **TODO: link**. The embeddings are OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`). |
| CelebA-B | `cbin/vectors.npz` | `python scripts/data/prepare_data.py cbin` (from CelebA) |
| SIFT1M | `sift1m/vectors.npz`, `sift1m/metadata_V{5,10,20,50}.npz` | see below |
| SIFT10M | `sift10m/vectors.npz` | see below |
| SIFT1B | `bigann_1b/` | see "Billion-scale runs" |
| Synthetic 2 | `synthetic/` | see "Billion-scale runs" |

To build SIFT1M and SIFT10M:
1. Download `bigann_base.bvecs.gz` from the [BIGANN benchmark](http://corpus-texmex.irisa.fr/) to `data/`.
2. Run:

```bash
python scripts/data/prepare_bigann.py --n 1000000  --out_dir data/bigann_1m
python scripts/data/prepare_bigann.py --n 10000000 --out_dir data/bigann_10m
python scripts/data/prepare_data.py sift1m variants sift10m
```

`prepare_bigann.py` draws three attributes with 3, 4 and 3 values, independently of the vectors. The `variants` step adds the Zipf-distributed attribute draws with |V_j| = 5, 10, 20 and 50 used for the partition-scalability table.

## Reproducing the experiments

```bash
# 1. workloads: 200 ground-truth-feasible queries per configuration (seed 10), in data/<name>/icde/
python scripts/runjobs.py jobs/00_workloads.txt --workers 4

# 2. experiments; raw results go to outputs/<dataset>/results_ranged_<workload>/<method>/
python scripts/runjobs.py jobs/01_q1_single_index.txt --workers 4
python scripts/runjobs.py jobs/02_q2_postprocessing.txt --workers 1   # timings of Section V-C: one run at a time
python scripts/runjobs.py jobs/03_q3_framework.txt --workers 4
python scripts/runjobs.py jobs/04_q5_partitions.txt --workers 4

# 3. evaluate every run into one table (overwrites results/all_runs.csv)
python scripts/collect.py

# 4. figures, numbers and tables
python scripts/figures.py           # results/figs/exp_*.pdf
python scripts/report_numbers.py    # default settings, build-cost and partition-scalability rows, single index vs. framework
python scripts/report_post.py       # post-processing times, solver agreement, greedy failures
python scripts/depth_analysis.py    # rank of the farthest member of the fair optimum (results/depth.csv)
```

Each line of a job file is one command. `runjobs.py` logs every job to `logs/<hash>.log` and skips jobs that have already succeeded, so an interrupted run can be resumed. The largest run, Alg-Fair-LSH on SIFT10M with *ℓ* = 64, takes about 2.4 h to build and peaks at about 60 GB of RAM.

`results/` already holds the evaluated results behind the paper (`all_runs.csv`, `depth.csv`, `billion_strict.csv`) and the figures made from them. `python scripts/figures.py quality postfilter pareto billion` redraws those figures without running any experiment. The post-processing figure (`post`) also reads the raw results in `outputs/`, so it needs step 2.

| Paper | Jobs | Script |
|---|---|---|
| Q1: single-index baselines (Fig. `exp_postfilter`), median ranks | `01` | `figures.py postfilter`, `depth_analysis.py` |
| Q2: post-processing (Fig. `exp_post`) | `02` | `figures.py post`, `report_post.py` |
| Q3: quality (Fig. `exp_quality`), recall vs. query time (Fig. `exp_pareto`) | `01`, `03` | `figures.py quality pareto` |
| Q3: candidates, success, recall and time of the single-index baselines vs. the framework on FairFace and CelebA (abstract, Sections I and V-D) | `01`, `03` | `report_numbers.py` |
| Q4: build time and index size (build-cost table) | `01`, `03` | `report_numbers.py` |
| Q5: number of partitions (partition-scalability table) | `04` | `report_numbers.py` |
| Q5: billion scale (Fig. `exp_billion`) | `scripts/billion/` | `figures.py billion` |
| Share of feasible random queries (Setup) | `00` | `data/<name>/icde/*_stats.json` |

## Billion-scale runs

These runs predate the rest of the setup: Alg-Fair-LSH with *μ* = 2, *k* = 5, CBC, and per-partition hash functions. The vectors are memory-mapped and the index is kept on disk. The vectors alone take 512 GB (SIFT1B) or 128 GB (Synthetic 2).

```bash
python scripts/data/prepare_bigann.py --n 1000000000 --out_dir data/bigann_1b
python generate_queries_billion.py --dataset bigann_1b --k 5 --m 3
bash scripts/billion/run_sift1b_sweep.sh

python scripts/data/generate_synthetic.py --n 1000000000 --d 32 --out_dir data/synthetic
python generate_queries_ranged.py --dataset synthetic --k 5 --m 3
bash scripts/billion/run_synthetic_sweep.sh
```

The sweep scripts build the index for *ℓ* = 4 to 64 and evaluate the queries with `evaluate_ranged.py`. `results/billion_strict.csv` holds the numbers shown in the paper (50 queries on SIFT1B, 200 on Synthetic 2).

## License

Apache 2.0 (see `LICENSE`).
