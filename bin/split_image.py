#!/usr/bin/env python

import os
from tifffile import TiffWriter
import argparse

from utils import read_tiff_orion
import psutil


def strip_gen(arr, y, y_size, x_chunk=4096):
    for channel in range(arr.shape[0]):
        for x_cur in range(0, arr.shape[2], x_chunk):
            yield arr[channel, y:y+y_size, x_cur: x_cur+x_chunk]


def split_img(img_path, out_dir, height=224, overlap=0.1, memory=0, scaling=1):
    """Will split an image into height x image_width crop (with some overlap) to get a better memory footprint """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    img_zarr, metadata = read_tiff_orion(img_path)

    if out_dir is None:
        out_dir = os.path.dirname(img_path)
    ch, total_height, total_width = img_zarr.shape

    print(f"memory: {memory}, {height=} shape={img_zarr.shape}")

    if memory or not height:
        computed_max_height = int(int(memory) * scaling / (img_zarr.dtype.itemsize * 8 * total_width * (ch+2)))
        #                     memory_per_cpu * re-scaling of the image / (size_of_pixel_in_bytes * nb_bit_per_byte * width * channel + 2 to get some margin)
        height = min(height, computed_max_height) if height else computed_max_height

    strip_shape = list(img_zarr.shape)

    for i, cur_height in enumerate(range(0, total_height, int(height * (1 - overlap))), 1):
        out_path = os.path.join(out_dir, img_name + f"_{cur_height}" + ext)
        strip_shape[1] = height if cur_height+height < total_height else total_height - cur_height
        with TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
            #tmp_arr = img_zarr[:, cur_height: cur_height+height, :]
            metadata.pix.size_y = strip_shape[1] # last one is not height unless total_height % height = 0
            tiff_out.write(
                data=strip_gen(img_zarr, cur_height, strip_shape[1]),
                shape=strip_shape,
                **metadata.to_dict()
            )
        if not (i % 10):
            with open(f'log_{i}.txt', 'a') as out:
                out.write("\nbefore : \n")
                out.write(str(psutil.virtual_memory()))
                out.write(f"\n{locals()}\n\n{img_zarr.info}")
#            del tmp_arr
#            img_zarr, metadata = read_tiff_orion(img_path) # need this to clear memory usage (I hope)

#            with open(f'log_{i}.txt', 'a') as out:
#                out.write("\nafter : \n")
#                out.write(str(psutil.virtual_memory()))
#                out.write(f"\n{locals()}\n\n{img_zarr.info}")
    print(i) # needed for nextflow to be aware of the number of file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str, required=True, help="Image Path to split")
    parser.add_argument('--out', type=str, required=False, help="Output directory for resulting images")
    parser.add_argument('--height', type=int, required=False, help="height of tiles")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="percentage of overlap for tiles")
    parser.add_argument('--memory', type=float, required=False, default=0, help="memory size available for each crop")
    parser.add_argument('--scaling', type=float, required=False, default=1, help="scaling of image before seg (with another diameter than 30, image is rescaled beforehand)")
    args = parser.parse_args()
    rtyu = dict(img_path=args.file_in, out_dir=args.out, height=args.height, overlap=args.overlap, memory=args.memory, scaling=args.scaling)
    print(rtyu)
    split_img(img_path=args.file_in, out_dir=args.out, height=args.height, overlap=args.overlap, memory=args.memory, scaling=args.scaling)