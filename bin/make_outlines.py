#!/usr/bin/env python


import argparse
import tifffile
import numpy as np
import os
import cv2
from PIL import Image
from scipy.ndimage import find_objects
# ===! VULNERABILITY !===
Image.MAX_IMAGE_PIXELS = None # raise DOSbombing error when too many pixels

def to_8int(arr, method="median_unbiased", percentile=[0.1,99.9], channel_axis=0):
    min_, max_ = np.percentile(arr, percentile, axis=[a for a in range(len(arr.shape)) if a != channel_axis], 
                               keepdims=True, method=method)
    new_arr = ((arr - min_) * 255 / max_)
    new_arr[new_arr < 0] = 0
    new_arr[new_arr > 254] = 254
    return new_arr.astype('uint8')

def create_outline_mask(masks):
    # see cellpose code source to know how to do it
    outlines = np.zeros(masks.shape, np.uint8)
    slices = find_objects(masks.astype(int))
    for i, slice in enumerate(slices):
        if slice is not None:
            sr, sc = slice
            mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
            vr, vc = pvr + sr.start, pvc + sc.start 
            outlines[vr, vc] = 255
    return outlines

def make_outline(merged_file, png_file, mask_path, out_path, nuclei_channel=0, cyto_channel=1):
    tiff = tifffile.TiffFile(merged_file) # blue = nuclei = 0, green = cyto = 1
    channel_to_keep = [int(nuclei_channel)]
    if int(cyto_channel) >= 0:
        channel_to_keep.append(int(cyto_channel))
    result = np.moveaxis(to_8int(tiff.series[0].asarray()[channel_to_keep, ...]), 0, -1).copy() # tiff are CYX

    if png_file is not None:
        png = np.array(Image.open(png_file))
        outline = np.zeros_like(png[..., [0]])
        outline[(png[..., 0] == 255) & np.all(png[..., [1,2]] == 0, axis=2)] = 255
    else:
        mask = np.array(Image.open(mask_path))
        outline = create_outline_mask(mask)
        print(np.unique(outline))
    print(f"result = {result.shape}, outline = {outline.shape}")
    result = np.append(outline[..., np.newaxis], np.flip(result, axis=2), axis=2) 
    return tifffile.imwrite(out_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_tiff', type=str, required=True, help="Tiff file with at least two channels (nuclei and cyto)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--png_outline', type=str, help="Png outline")
    group.add_argument('--mask', type=str, help="mask (output of segmentation) in tiff format")
    parser.add_argument('--out', type=str, required=False, help="Output filepath")
    parser.add_argument('--nuclei', type=str, required=False, default=0, help="index of nuclei channel in tiff file")
    parser.add_argument('--cyto', type=str, required=False, default=1, help="index of cytoplasm channel in tiff file")
    args = parser.parse_args()

    merge_tiff = vars(args)['merge_tiff']
    out_path = args.out
    if out_path is None:
        tokens = os.path.basename(merge_tiff).split(os.extsep)
        if "_merged" in tokens: tokens = tokens.rsplit('_', 1)[0]
        if len(tokens) < 2:       stem = merge_tiff
        elif tokens[-2] == "ome": stem = os.extsep.join(tokens[0:-2])
        else:                     stem = os.extsep.join(tokens[0:-1])
        out_path = stem + "_clear_outlines.tiff"

    make_outline(merge_tiff, args.png_outline, args.mask, out_path, args.nuclei, args.cyto)