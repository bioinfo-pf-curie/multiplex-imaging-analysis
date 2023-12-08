#!/usr/bin/env python

import os
from tifffile import TiffWriter
import argparse

from utils import read_tiff_orion


def split_img(img_path, out_dir, height=224, overlap=0.1, memory=0):
    """Will split an image into height x image_width crop (with some overlap) to get a better memory footprint """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    img_zarr, metadata = read_tiff_orion(img_path)

    if out_dir is None:
        out_dir = os.path.dirname(img_path)
    ch, total_height, total_width = img_zarr.shape

    if memory or not height:
        computed_max_height = int(int(memory) / (img_zarr.dtype.itemsize * 8 * total_width * (ch+2)))
        #                     memory_per_cpu / (size_of_pixel_in_bytes * nb_bit_per_byte * width * channel + 2 to get some margin)
        height = min(height, computed_max_height) if height else computed_max_height

    for cur_height in range(0, total_height, int(height * (1 - overlap))):
        out_path = os.path.join(out_dir, img_name + f"_{cur_height}" + ext)
        with TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
            tmp_arr = img_zarr[:, cur_height: cur_height+height, :]
            metadata.pix.size_x = tmp_arr.shape[1] # last one is not height unless total_heigh % height = 0
            tiff_out.write(
                data=tmp_arr,
                shape=tmp_arr.shape,
                **metadata.to_dict()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str, required=True, help="Image Path to split")
    parser.add_argument('--out', type=str, required=False, help="Output directory for resulting images")
    parser.add_argument('--height', type=int, required=False, help="height of tiles")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="percentage of overlap for tiles")
    parser.add_argument('--memory', type=str, required=False, default=0, help="memory size available for each crop")
    args = parser.parse_args()

    split_img(img_path=args.file_in, out_dir=args.out, height=args.height, overlap=args.overlap, memory=args.memory)