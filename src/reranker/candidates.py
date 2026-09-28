import numpy as np

from .config import SOURCE_NAMES, SOURCE_TOP_K


def candidate_union(indices, row_idx, source_top_k=SOURCE_TOP_K):
    arrays = [
        np.asarray(
            indices[source][row_idx, :source_top_k],
            dtype=np.int32,
        )
        for source in SOURCE_NAMES
    ]

    return np.unique(np.concatenate(arrays))
