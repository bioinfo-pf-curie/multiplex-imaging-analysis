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


def compute_current_cell_id(block_info, mean_cell_area=600):
    """
    From chunk info (from dask map_block, see https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html)
    compute the current cell id based on chunk position, mean chunk size and mean cell area

    parameters
    ----------

    block_info: dict of dict
        it is constructed as in : 
        {0: {'shape': (1000,),
             'num-chunks': (10,),
             'chunk-location': (4,),
             'array-location': [(400, 500)]},
         None: {'shape': (1000,),
                'num-chunks': (10,),
                'chunk-location': (4,),
                'array-location': [(400, 500)],
                'chunk-shape': (100,),
                'dtype': dtype('float64')}}
        
    mean_cell_area: int
        mean cell area, can be infered on the image but a value of 600 (default) is satisfying for now

    return
    ------

    current_cell_id: int
        index for the current chunk to start from
    """
    try: # try except block is mandatory because dask will try with empty block to guess output dtype
        _, row_chunk, col_chunk = block_info[0]['chunk-location']
        _, row_total, col_total = block_info[0]['num-chunks']
        _, row_shape, col_shape = block_info[0]['shape']
        mean_cells_per_chunk = (row_shape / row_total) * (col_shape / col_total) / mean_cell_area
        current_cell_id = int((row_chunk +  col_chunk * row_total) * mean_cells_per_chunk)
    except:
        current_cell_id = 1
    return current_cell_id
