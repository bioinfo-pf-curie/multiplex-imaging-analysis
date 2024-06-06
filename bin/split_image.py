#!/usr/bin/env python

import os
from tifffile import TiffWriter
import argparse

from utils import read_tiff_orion, min_max_norm
import numpy as np


def split_img(img_path, out_dir, height=224, overlap=0.1, memory=0, normalize=False):
    """Will split an image into height x image_width crop (with some overlap) to get a better memory footprint """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    img_zarr, metadata = read_tiff_orion(img_path, zarr_mode='a', mode='r+b')

    if out_dir is None:
        out_dir = os.path.dirname(img_path)
    ch, total_height, total_width = img_zarr.shape

    if memory or not height:
        computed_max_height = int(int(memory) / (img_zarr.dtype.itemsize * 8 * total_width * (ch+2)))
        #                     memory_per_cpu / (size_of_pixel_in_bytes * nb_bit_per_byte * width * channel + 2 to get some margin)
        height = min(height, computed_max_height) if height else computed_max_height

    if normalize:
        from dask import array as da
        arr = da.from_zarr(img_zarr)
        for i in [0,1]:
            img_min, img_max = da.percentile(arr[i].flatten(), [1, 99]).compute()
            arr[i] = ((arr[i] - img_min) / (img_max - img_min))
        arr.to_zarr(img_zarr, overwrite=True, compute=True)
    # l   ow, high = np.percentile(img_zarr, [1,99], axis=(1,2))

    for i, cur_height in enumerate(range(0, total_height, int(height * (1 - overlap))), 1):
        out_path = os.path.join(out_dir, img_name + f"_{cur_height}" + ext)
        with TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
            tmp_arr = img_zarr[:, cur_height: cur_height+height, :]
            metadata.pix.size_y = tmp_arr.shape[1] # last one is not height unless total_heigh % height = 0
            tiff_out.write(
                data=tmp_arr,
                shape=tmp_arr.shape,
                **metadata.to_dict()
            )
    print(i) # needed for nextflow to be aware of the number of file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str, required=True, help="Image Path to split")
    parser.add_argument('--out', type=str, required=False, help="Output directory for resulting images")
    parser.add_argument('--height', type=int, required=False, help="height of tiles")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="percentage of overlap for tiles")
    parser.add_argument('--memory', type=float, required=False, default=0, help="memory size available for each crop")
    args = parser.parse_args()

    split_img(img_path=args.file_in, out_dir=args.out, height=args.height, overlap=args.overlap, memory=args.memory, normalize=True)