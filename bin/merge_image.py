#!/usr/bin/env python

import os
import argparse
from tifffile import TiffFile, memmap, imwrite, imread
from numpy import newaxis, moveaxis
import zarr

def create_empty_img(out_path, input_img_path, example_img):
    """
        >>> imwrite('temp.ome.tif', shape=(8, 800, 600), dtype='uint16',
        ...         photometric='minisblack', tile=(128, 128),
        ...         metadata={'axes': 'CYX'})
        >>> store = imread('temp.ome.tif', mode='r+', aszarr=True)
        >>> z = zarr.open(store, mode='r+')
        >>> z
        <zarr.core.Array (8, 800, 600) uint16>
        >>> z[3, 100:200, 200:300:2] = 1024
        >>> store.close()
    """
    tiff = TiffFile(input_img_path) 
    # imwrite(out_path, shape=(*tiff.series[0].shape[1:], example_img.shape[2]), dtype=example_img.dtype, 
    #         photometric=tiff.pages[0].photometric, tile=(128, 128), resolution=(
    #                 tiff.pages[0].tags["XResolution"].value,
    #                 tiff.pages[0].tags["YResolution"].value,
    #                 tiff.pages[0].tags["ResolutionUnit"].value),
    #         metadata={'axes': 'CYX'})
    # maybe more metadata can be transferred...
    return memmap(out_path, 
                  shape=(example_img.shape[0], *tiff.series[0].shape[1:]), 
                  dtype=example_img.dtype, 
                  bigtiff=True, ome=True,
                  software=tiff.pages[0].software,
                  resolution=(
                    tiff.pages[0].tags["XResolution"].value,
                    tiff.pages[0].tags["YResolution"].value,
                    tiff.pages[0].tags["ResolutionUnit"].value),
                  photometric=tiff.pages[0].photometric,
                  contiguous=True)

def merge_image(img_path_list, out_path, input_img_path):
    out_img = None
    suffix = ["_merged_cp_masks", "_merged_clear_outlines"] # todo : probably better with regexp...
    for img_path in img_path_list:
        img_name, ext = os.path.splitext(os.path.basename(img_path))

        for sfx in suffix:
            img_name = img_name.removesuffix(sfx)

        img_name, cur_height = img_name.rsplit('_', 1)
        cur_height = int(cur_height)

        img = TiffFile(img_path)
        img = img.series[0][0].asarray()

        if len(img.shape) == 2:
            img = img[..., newaxis]

        img = moveaxis(img, 2, 0)

        if out_img is None:
            # create_empty_img(out_path, input_img_path, img)
            # store = imread(out_path, mode='r+', aszarr=True)
            # out_img = zarr.open(store, mode='r+')
            out_img = create_empty_img(out_path, input_img_path, img)

        out_img[:, cur_height:cur_height+img.shape[1], :] = img
    # store.close()
    out_img.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    args = parser.parse_args()

    merge_image(vars(args)['in'], args.out, args.original)
