#!/usr/bin/env python

import argparse
import tifffile
from pathlib import Path

from merge_masks import solve_conflicts, recreate_mask, extract_cell_geoms
from utils import get_current_height, OmeTifffile
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, help="Not used")
    args = parser.parse_args()
    t0 = time.process_time()
    list_npy = vars(args)['in']

    original_tiff = tifffile.TiffFile(args.original)
    original_shape = original_tiff.series[0].shape[1:]
    out_path = f"{Path(args.original).stem}_masks.tiff"
    total_cells = []
    t1 = time.process_time()
    t2 = []

    for i, tile in enumerate(list_npy):
        cur_height = get_current_height(tile)
        img = tifffile.imread(tile)
        total_cells += extract_cell_geoms(img, transform=(0, cur_height))
        t2.append(time.process_time() - t1)

    unique_cells = solve_conflicts(total_cells, threshold=0.1)
    t3 = time.process_time()
    result = recreate_mask(unique_cells, original_shape, 1)
    t4 = time.process_time()
    try:
        metadata = OmeTifffile(original_tiff.pages[0])
    except: 
        metadata = OmeTifffile()
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = result.dtype
    metadata.update_shape(result.shape)

    kwargs = metadata.to_dict()
    t5 = time.process_time()
    tifffile.imwrite(out_path, result, bigtiff=True, shaped=False, **kwargs)
    t6 = time.process_time()
    print(f'reading original : {t1-t0:.02f}, iterations : {t2}, solve conflict : {t3 - (t2[-1] + t1):.02f}, recreate : {t4-t3:.02f}, metadata : {t5-t4:.02f}, write : {t6-t5:.02f}')

""" #test to debug...
import tifffile
from merge_masks import merge_masks
import numpy as np

test_path = "orion/fichier_test/instanseg_test/"
list_npy = [tifffile.imread(test_path + f"img{i}.tiff") for i in range(5)]

result = np.zeros((1024,1024))
img = np.pad(list_npy[0], [(0,768), (0,0)])
a = merge_masks([result, img], threshold=0.1, remap=False)

"""