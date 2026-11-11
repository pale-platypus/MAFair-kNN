from .l2lsh_cartesian import L2LSHCartesian
from .l2ivf_cartesian import L2IVFCartesian
from .exact_cartesian import ExactCartesian
from .global_postfilter import GlobalPostFilter
from .filter_single import FilterIVF, FilterLSH
from .sieve_cartesian import SIEVECartesian
from .filter_diskann import FilterDiskANN

# --index name -> class (paper name in the comment)
indexes = {
    "l2lsh_cartesian": L2LSHCartesian,      # Alg-Fair-LSH
    "l2ivf_cartesian": L2IVFCartesian,      # Alg-Fair-IVF
    "sieve_cartesian": SIEVECartesian,      # Alg-Fair-HNSW
    "exact_cartesian": ExactCartesian,      # Alg-Fair-Exact
    "filter_diskann": FilterDiskANN,        # HNSW-baseline
    "filter_lsh": FilterLSH,                # LSH-baseline
    "filter_ivf": FilterIVF,                # IVF-baseline
    "global_postfilter": GlobalPostFilter,  # Global+PF
}


def build_index(args):
    if indexes.get(args.index, None) is not None:
        return indexes[args.index]
    else:
        raise NameError(f"Undefined indexing method for {args.index}!")
