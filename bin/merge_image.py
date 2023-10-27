#!/usr/bin/env python

import os
from tifffile import TiffWriter, memmap

from utils import read_tiff_orion

def create_empty_img(out_path, input_img_path):
    tiff = tifffile.TiffFile(input_img_path)
    # maybe more metadata can be transferred...
    return memmap(out_path, shape=tiff.series[0].shape, dtype=tiff.dtype, bigtiff=True)

def merge_image(img_path_list, out_path, input_img_path):
    out_img = create_empty_img(out_path, input_img_path)
    for img_path in img_path_list:
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_name, cur_height = img_name.removesuffix("_merged_cp_masks").rsplit('_', 1)
        cur_height = int(cur_height)
        img, metadata = read_tiff_orion(img_path)
        out_img[:, cur_height:cur_height+img.series[0].shape[1], :] = img.series[0]
    out_img.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="comma separated list of Image Path (cropped)")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    args = parser.parse_args()

    merge_image(var(args)['in'], args.out, args.original)
