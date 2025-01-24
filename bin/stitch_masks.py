#!/usr/bin/env python

import argparse
import tifffile
from pathlib import Path
import numpy as np
import fastremap

from merge_masks import merge_masks
from utils import get_current_height, OmeTifffile


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
    # tmp_path = ".tmp_masks.npy"
    # result = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.uint32, shape=original_shape)
    # result = tifffile.memmap(out_path, dtype="uint32", shape=original_shape, mode='w+')
    # px_overlap = int(original_shape[0] * args.overlap)
    result = np.zeros(original_shape)

    # resulting_shape = np.zeros(original_shape)
    # result = merge_masks([resulting_shape] + list_npy, transform=[(0,0)] + [(0, get_current_height(tile)) for tile in list_npy])

    for i, tile in enumerate(list_npy):
        print(f"in tile : '{tile}'")
        cur_height = get_current_height(tile)
        img = tifffile.imread(tile)

        # starting_point = max(cur_height - px_overlap, 0)
        # ending_point = min(cur_height + img.shape[0] + px_overlap, original_shape[0])

        # print((starting_point, ending_point))
        # print(f"value before : {result[starting_point: ending_point, :].shape}")

        # augmented_img = np.pad(img.astype('uint32'), ((starting_point and px_overlap, (ending_point != original_shape[0]) and px_overlap), (0,0)))
        # print(f"img before : {augmented_img.shape}")

        # r = merge_masks([result[starting_point:ending_point, :], augmented_img], chunk_size=8192, remap=False) # , transform=[(0,0), (0, starting_point and px_overlap)]
        result = merge_masks([result, img], chunk_size=8192, transform=[(0,0), (0, cur_height)], remap=False) # 

        print(result.max())
        # result[starting_point:ending_point, :] = r

        if not i % 10: # flush every ten file (~10GB)
            result.flush()
            # reload memmap each time else it will accumulate in memory
            result = np.lib.format.open_memmap(tmp_path, mode="r+")
    result.flush()

    fastremap.renumber(result, in_place=True)

    metadata = OmeTifffile(original_tiff.pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = result.dtype
    metadata.update_shape(result.shape)

    kwargs = metadata.to_dict()

    tifffile.imwrite(out_path, result, bigtiff=True, shaped=False, **kwargs)
    # Path(tmp_path).unlink()
