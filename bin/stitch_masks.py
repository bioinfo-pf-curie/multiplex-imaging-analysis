#!/usr/bin/env python

import argparse
import tifffile
from pathlib import Path

from merge_masks import merge_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="value of overlap used for splitting images")
    args = parser.parse_args()

    list_npy = vars(args)['in']

    original_tiff = tifffile.TiffFile(args.original)
    original_shape = original_tiff.series[0].shape[1:]

    result = tifffile.memmap(f"{Path(args.original).stem}_masks.tiff", dtype="uint32", shape=(1, *original_shape))
    result[...] = merge_masks(list_npy, overlap=args.overlap, chunk_size=8192, threshold=args.threshold)

    