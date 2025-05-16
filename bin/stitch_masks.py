#!/usr/bin/env python

import argparse
from geopandas import GeoDataFrame
import tifffile
from pathlib import Path
from utils import get_current_height, OmeTifffile
from numpy import array
import dask
import numpy as np
import datashader as ds

from merge_masks import solve_conflicts, recreate_mask, extract_cell_geoms

def rasterize(cells, height, width):
    cnv = ds.Canvas(plot_height=height, plot_width=width)
    return cnv.polygons(cells, "geometry", agg=ds.count())

def create_mask_chunk(cells_in_chunk, chunk_pos, chunk_size, threshold=0.1):
    unique_cells = solve_conflicts(cells_in_chunk, threshold=threshold)
    unique_cells = GeoDataFrame(geometry=unique_cells,index=cells_in_chunk.index[:len(unique_cells)])
    unique_cells.geometry = unique_cells.geometry.translate(xoff=-chunk_pos[0], yoff=chunk_pos[1])
    x, y = chunk_pos
    # res = rasterize(unique_cells, *chunk_size)
    res = recreate_mask(unique_cells.geometry.values, chunk_size, unique_cells.index)
    with open(f"log_pos_{x}_{y}.txt", "a") as log:
        log.write(f"{res.shape} and {np.unique(res)}\n\n")
    return res

def stitch_mask(tiles_names, tiles_height, original_shape, out_path, chunk_size=2048):
    """
    Create spatial data for each tile.
    """
    total_cells = []
    for tile, cur_height in zip(tiles_names, tiles_height):
        # cur_height = get_current_height(tile)
        img = tifffile.imread(tile)
        total_cells.extend(extract_cell_geoms(img, transform=(0, cur_height)))
    gdf = GeoDataFrame(geometry=total_cells, index=range(1, len(total_cells)+1))
    # z1 = zarr.create_array(store=out_path, shape=original_shape, chunks=(2048, 2048), dtype='uint32')

    def gen_chunks():
        for x in range(0, original_shape[0], chunk_size):
            for y in range(0, original_shape[1], chunk_size):
                chunk = gdf.cx[x:x+chunk_size, y:y+chunk_size]
                if not chunk.empty:
                    ert = create_mask_chunk(chunk.geometry, (x,y), (chunk_size, chunk_size), threshold=0.1).astype('uint32')
                    tifffile.imwrite(f"res_pos_{x}_{y}.tiff", ert)
                    yield ert
                else:
                    yield np.zeros((chunk_size, chunk_size), dtype='uint32')

    with tifffile.TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
        tiff_out.write(
            data=gen_chunks(),
            shape=original_shape,
            tile=(chunk_size, chunk_size),
            dtype='uint32'
        )
    # dask.compute(*resolving_chunks)
    # output = da.from_zarr(z1)
    # da.map_overlap(generate_mask_chunk, output, depth=100, boundary=0, dtype='uint32')
    # return spatial_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, help="Not used")
    args = parser.parse_args()
    list_npy = vars(args)['in']

    original_tiff = tifffile.TiffFile(args.original)
    original_shape = original_tiff.series[0].shape[1:]
    out_path = f"{Path(args.original).stem}_masks.tiff"
    total_cells = []
    patch_indices = []

    for i, tile in enumerate(sorted(list_npy)):
        cur_height = get_current_height(tile)
        img = tifffile.imread(tile)
        new_cells = extract_cell_geoms(img, transform=(0, cur_height))
        patch_indices += [i] * len(new_cells)
        total_cells += new_cells
        # total_cells = solve_conflicts(total_cells + new_cells, patch_indices=[0] * len(total_cells) + [1] * len(new_cells), threshold=0.1)

    unique_cells = solve_conflicts(total_cells, patch_indices=array(patch_indices), threshold=0.1)
    result = recreate_mask(unique_cells, original_shape, 1)

    try:
        metadata = OmeTifffile(original_tiff.pages[0])
    except: 
        metadata = OmeTifffile()
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = result.dtype
    metadata.update_shape(result.shape)

    kwargs = metadata.to_dict()

    tifffile.imwrite(out_path, result, bigtiff=True, shaped=False, **kwargs)

""" #test to debug...
import tifffile
from merge_masks import merge_masks
import numpy as np

test_path = "orion/fichier_test/instanseg_test/"
list_npy = [tifffile.imread(test_path + f"img{i}.tiff") for i in range(5)]

result = np.zeros((1024,1024))
img = np.pad(list_npy[0], [(0,768), (0,0)])
a = merge_masks([result, img], threshold=0.1, remap=False)

import shapely
from stitch_masks import stitch_mask
test_path = '/data/users/mcorbe/orion/fichier_test/instanseg_test/'
list_msk = [test_path + f"img_{i}.tiff" for i in [0,250,500,750,1000]]
stitch_mask(list_msk, (0,250,500,750,1000), (1024,1024), "result_mask_new_test.tiff", chunk_size=256)

"""