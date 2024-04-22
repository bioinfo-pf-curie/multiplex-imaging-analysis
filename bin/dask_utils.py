#!/usr/bin/env python

import pandas as pd

# for each grp (!= 0) return value most frequent (unless its 0)
# This function will try to stitch masks together when there is contact between two of them 
def adjacent_mask_filter(grp):
    if grp.name:
        try:
            v = int(grp.mode().values)
        except TypeError:
            return
        if v:
            return v


def correct_edges_inplace(masks, chunks_size):
    """
    Take the limit of each chunk and stitch together corresponding mask

    parameters
    ----------

    masks: int32, 2D array
        masks to modify

    chunks_size: tuple of int
        size of the chunk

    return
    ------

    nothing, modifying masks inplace
    """
    for dim in [0,1]:
        for limit in range(chunks_size[dim], masks.shape[dim], chunks_size[dim]):
            # np.take allow to generalize between dimension (otherwise I should have written masks[limit-1] and masks[:, limit-1])
            replace = pd.DataFrame({0: masks.take(limit-1, axis=dim), 1: masks.take(limit, axis=dim)}, dtype=int)\
                .groupby(0, sort=False)[1]\
                .apply(adjacent_mask_filter)\
                .dropna().astype(int)
            sub_masks = masks[:, limit-256:limit+1] if dim else masks[limit-256:limit+1]
            # take make a copy
            for k, v in replace.items():
                sub_masks[sub_masks == k] = v
            sub_masks = None # garbage collected

