#!/usr/bin/env python


import argparse
import tifffile
import numpy as np
import os
import cv2
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import find_objects
import zarr
from pandas import read_csv

from utils import OmeTifffile, _tile_generator

# ===! VULNERABILITY !===
Image.MAX_IMAGE_PIXELS = None # raise DOSbombing error when too many pixels

def to_8int(arr, method="median_unbiased", percentile=[0.1,99.9], channel_axis=0):
    """
    little helper to transform a 16bits image into a 8bit one.
    It will normalize data beforehand between percentile.
    
    Parameters
    ----------
    
    arr: np.array
        image data

    method: str
        method to compute percentile value

    percentile: list of float
        bottom and top percentile value. Outside this range value are set at 0 and 254

    channel_axis: int
        index of the dimension use by channels. By default in our case its always 0.
    
    Return
    ------
        
        new_arr: np.array
           image data with dtype = uint8
    """
    min_, max_ = np.percentile(arr, percentile, axis=[a for a in range(len(arr.shape)) if a != channel_axis], 
                               keepdims=True, method=method)
    new_arr = ((arr - min_) * 255 / max_)
    np.clip(new_arr, 0, 254, new_arr)
    return new_arr.astype('uint8')

def create_outline_mask(masks):
    """
    From an array of masks create a new array with the outline of each mask.
    Copied from cellpose code source (https://github.com/MouseLand/cellpose/blob/main/cellpose/utils.py#L191)
    
    Parameters
    ----------
    
    masks: np.array
        array of masks
        
    Return
    ------
    
    outlines: np.array
        array the size of underline image with only outlines in it.
        
    """
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

def make_outline(merged_file, png_file, mask_path, out_path, nuclei_channel=0, cyto_channel=1, all_channels=False, channel_info=None):
    """
    This function accept two sort of input to get outlines.
    First case a png file with red channel = outlines, green channel = cyto and blue = nucleus (can be an options in cellpose parameters)
    Second is a mask tiff file which will be converted to outlines.
    
    It will output a png file (exactly like the one in first case) with red = outline, blue = nuclei_channel from merged_file and green = cyto_channel
    If all_channels is True, cyto_channel will be created by merging all other channel from merged_file.

    Parameters
    ----------

    merged_file: np.array
        original image
    
    png_file: np.array or None
        image to get outline from. can be set to None if mask_path is set.

    mask_path: str or Path
        path to mask image (PIL is used to open image)
    
    out_path: str or Path
        path to output file
    
    nuclei_channel: int
        index for the nuclei channel
    
    cyto_channel: int
        index for the cyto channel (its ignored if all_channels = True)

    all_channels: bool
        Flag to merge all other channel into a new one for the cyto channel.

    Return
    ------

    None
    """
    if png_file is not None:
        png = np.array(Image.open(png_file))
        outline = np.zeros_like(png[..., 0])
        outline[(png[..., 0] == 255) & np.all(png[..., [1,2]] == 0, axis=2)] = 255
    else:
        try:
            mask = np.array(Image.open(mask_path))
        except UnidentifiedImageError:
            mask = tifffile.imread(mask_path)
        outline = create_outline_mask(mask)

    tiff = tifffile.TiffFile(merged_file) # blue = nuclei = 0, green = cyto = 1
    metadata = OmeTifffile(tiff.pages[0])
    metadata.add_channel_metadata(channel_name="Outline")

    if not all_channels:
        channel_to_keep = [int(nuclei_channel)]
        try:
            if int(cyto_channel) >= 0:
                channel_to_keep.append(int(cyto_channel))
        except ValueError:
            pass

        result = np.moveaxis(to_8int(tiff.series[0].asarray()[channel_to_keep, ...]), 0, -1).copy() # tiff are CYX
        result = np.append(outline[..., np.newaxis], np.flip(result, axis=2), axis=2)
        metadata.dtype = result.dtype
        return tifffile.imwrite(out_path, result)#, bigtiff=True, shaped=False, **metadata.to_dict())
    else:
        if channel_info is not None:
            try:
                channel_csv = read_csv(channel_info)
                if (len(channel_csv) + 1) == len(metadata.pix.channels):
                    for i, channel in enumerate(channel_csv["marker_name"]):
                        if channel:
                            metadata.pix.channels[i].name = channel
            except:
                raise

        result = zarr.open(tiff.series[0].aszarr())
        c, x, y = tiff.series[0].shape

        def tile_gen(original, outline, c, x, y, chunk_size=(256,256)):
            for c_cur in range(c):
                yield from _tile_generator(original, c_cur, x, y, *chunk_size)
            for tile in _tile_generator(outline, None, x, y, *chunk_size):
                yield np.squeeze(tile).astype(original.dtype)

        with tifffile.TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
            tiff_out.write(
                data=tile_gen(result[0] if tiff.series[0].is_pyramidal else result, outline, c=c, x=x, y=y), 
                shape=[c+1, x, y], 
                tile=(256, 256), 
                **metadata.to_dict(shape=[x, y])
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge-tiff', dest="merge_tiff", type=str, required=True, help="Tiff file with at least two channels (nuclei and cyto)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--png-outline', type=str, dest="png_outline", help="Png outline")
    group.add_argument('--mask', type=str, help="mask (output of segmentation) in tiff format")
    parser.add_argument('--out', type=str, required=False, help="Output filepath")
    parser.add_argument('--nuclei', type=str, required=False, default=0, help="index of nuclei channel in tiff file")
    parser.add_argument('--cyto', type=str, required=False, default=1, help="index of cytoplasm channel in tiff file")
    parser.add_argument('--all-channels', required=False, dest="all_channels", action='store_true', help="if selected will use original image and append outline to it")
    parser.add_argument('--channel-info', dest="channel_info", type=str, required=False, help="path of csv file with correct channel name")
    args = parser.parse_args()

    merge_tiff = vars(args)['merge_tiff']
    out_path = args.out
    if out_path is None:
        tokens = os.path.basename(merge_tiff).split(os.extsep)
        if "_merged" in tokens: tokens[0] = tokens[0].rsplit('_', 1)[0]
        if len(tokens) < 2:       stem = merge_tiff
        elif tokens[-2] == "ome": stem = os.extsep.join(tokens[0:-2])
        else:                     stem = os.extsep.join(tokens[0:-1])
        out_path = stem + "_outlines.tiff"

    make_outline(merge_tiff, args.png_outline, args.mask, out_path, args.nuclei, args.cyto, args.all_channels, args.channel_info)