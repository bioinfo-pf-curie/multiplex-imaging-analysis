#!/usr/bin/env python

import argparse

from tifffile import TiffFile, imwrite
from deepcell_toolbox.deep_watershed import deep_watershed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="filename of mesmer output in tiff format")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--chunks', type=int, nargs=2, required=False, default=(1024, 1024), help="Size of chunk for dask")
    parser.add_argument('--overlap', type=int, required=False, default=60, help="Overlap (in pixel) for dask to perform computing of masks on chunks")
    args = parser.parse_args()

    mesmer_output = TiffFile(vars(args)['in'])
    label_img = deep_watershed(mesmer_output)

    # metadata = OmeTifffile(TiffFile(args.original).pages[0])
    # metadata.remove_all_channels()
    # metadata.add_channel_metadata(channel_name="masks")

    # metadata.dtype = mask_memmap.dtype

    imwrite(args.out, label_img, bigtiff=True, shaped=False)
