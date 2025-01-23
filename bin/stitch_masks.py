#!/usr/bin/env python

import argparse
import tifffile
from pathlib import Path

from merge_masks import merge_masks
from utils import get_current_height


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="value of overlap used for splitting images")
    args = parser.parse_args()

    list_npy = vars(args)['in']

    original_tiff = tifffile.TiffFile(args.original)
    original_shape = original_tiff.series[0].shape[1:]
    out_path = f"{Path(args.original).stem}_masks.tiff"
    result = tifffile.memmap(out_path, dtype="uint32", shape=original_shape, mode='w+')
    
    for i, tile in enumerate(list_npy):
        print(f"in tile : '{tile}'")
        cur_height = get_current_height(tile)
        img = tifffile.imread(tile)
        print((cur_height, cur_height + img.shape[0]))
        print(img.dtype)
        r = merge_masks([result[cur_height:cur_height + img.shape[0], :], img.astype('uint32')], chunk_size=8192)
        print(r.max())
        result[cur_height:cur_height + img.shape[0], :] = r

        if not i % 10: # flush every ten file (~10GB)
            result.flush()
            # reload memmap each time else it will accumulate in memory
            result = tifffile.memmap(out_path, dtype="uint32", shape=original_shape, mode="r+")
    result.flush()
    # result[...] = merge_masks(list_npy, overlap=args.overlap, chunk_size=8192)
