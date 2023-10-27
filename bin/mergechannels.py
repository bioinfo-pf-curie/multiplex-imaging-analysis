#!/usr/bin/env python

"""
Script for merging channels into one. 
It will be used for an approximation of a cytoplasm channel, as our channels are only specific to a cell subpopulation.
"""
from itertools import product
from pandas import read_csv
import numpy as np
import ome_types
import tifffile
import argparse
import os

from utils import read_tiff_orion

def tile_generator(arr, nuclei_chan, to_merge_chan, x, y, chunk_x, chunk_y, agg=np.max):
    for ci in [nuclei_chan, to_merge_chan]:
        for x_cur in range(0, x, chunk_x):
            for y_cur in range(0, y, chunk_y):
                tmp_arr = arr[ci, x_cur: x_cur + chunk_x, y_cur: y_cur + chunk_y]
                if ci != to_merge_chan:
                    yield tmp_arr
                else:
                    yield agg(tmp_arr, axis=0)
    tmp_arr = None # don't wait for next iteration to flush this


def merge_channels(in_path, out_path, nuclei_chan=0, channels_to_merge=None, chunk_size=(256,256), agg=np.max):
    """
    take an image on disk, load chunks of it and merged all channels (first dimension) into one by agg function (np.max or np.mean mostly).
    Make exception for first and second channel (by default) to not be merged

    Parameters
    ----------
    in_path : Path or Str
        path of image to be merged
    out_path : Path or Str
        path of output
    nuclei_chan : int
        First channel to be excluded from the merge but present in output
    channels_to_merge : list of int
        list of all channels to be included in the merge
    chunk_size : tuple of two int
        Size of the chunk to be loaded in-memory (default = (256,256))
    agg : function
        Function to be use to aggregate channels together (default np.max). 
        np.mean can also be used or any function that take an array of shape [:,*chunk_size] 
        and an axis argument to point the dimension of the merge (0)

    Returns
    -------
    None

    """
    if nuclei_chan in channels_to_merge:
        raise ValueError("There is conflict between channels to merge and nuclei channels")
    img_level, metadata = read_tiff_orion(in_path)

    if channels_to_merge is None:
        channels_to_merge = list(range(2, img_level.shape[0]))

    with tifffile.TiffWriter(out_path, ome=True, bigtiff=True) as tiff_out:
            tiff_out.write(
                data=tile_generator(img_level, nuclei_chan, channels_to_merge, 
                                    *img_level.shape[1:], *chunk_size, agg=agg),
                software=metadata.software,
                shape=(2, *img_level.shape[1:3]),
                #subifds=int(self.num_levels - 1),
                dtype=metadata.dtype,
                resolution=(
                    metadata.tags["XResolution"].value,
                    metadata.tags["YResolution"].value,
                    metadata.tags["ResolutionUnit"].value),
                tile=chunk_size,
                photometric=metadata.photometric,
                compression="adobe_deflate",
                predictor=True,
            )

def guess_channels_to_merge(img_path):
    """we want to keep the nuclei marker and 
    remove the autofluorescen channel before merging"""
    info = ome_types.from_tiff(img_path)
    channels_info = info.images[0].pixels.channels
    try:
        nuclei = [c for c in channels_info 
                if "hoechst" in c.name.lower() or 
                "dapi" in c.name.lower()][0]
        nuclei = channels_info.index(nuclei)
    except AttributeError:
        return 0, None
    except IndexError:
        nuclei = 0
        # most of the time its the first one
    to_merge = [i for i,c in enumerate(channels_info) 
                if nuclei != i and "af1" != c.name.lower()]
    return nuclei, to_merge

def parse_markers(img_path, markers_path):
    """Use of markers.csv mandatory file for mcmicro to point channel to be merge for helping segmentation"""
    segmentation_col_name = "segmentation"

    mrk = read_csv(markers_path)
    if segmentation_col_name in mrk.columns:
        mrk[segmentation_col_name] = mrk[segmentation_col_name].fillna(False).astype(bool)
        result = {idx: row["marker_name"] for idx, row in mrk.iterrows() if row[segmentation_col_name]}
    else:
        print(f"no column {segmentation_col_name} found in markers.csv... guessing channels")
        return guess_channels_to_merge(img_path)[1]
    
    # need to compare channel name with metadata to get order 
     
     

    # (or its the same in both and we dont care)
    return list(result.keys())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="Input Image Path")
    parser.add_argument('--out', type=str, required=False, help="Output merged image Path")
    parser.add_argument('--channels', type=str, required=False, default=None, 
                        help="comma separated list or file with channels index to be merged (by default use every channels except the first)")
    parser.add_argument("--nuclei-channels", type=str, required=False, default=0, help="index of nuclei channel")
    args = parser.parse_args()

    in_path = vars(args)['in']
    out_path = args.out
    if out_path is None:
        tokens = os.path.basename(in_path).split(os.extsep)
        if len(tokens) < 2:       stem = in_path
        elif tokens[-2] == "ome": stem = os.extsep.join(tokens[0:-2])
        else:                     stem = os.extsep.join(tokens[0:-1])
        out_path = stem + "_merged.tif"

    channels = vars(args)['channels']
    if channels is not None:
        try:
            channels = [int(c) for c in channels.split(',')]
        except ValueError:
            if os.path.exists(channels):
                channels = parse_markers(in_path, channels)
            else:
                raise ValueError('Wrong format for channels')
    else:
        _, channels = guess_channels_to_merge(in_path)
    merge_channels(in_path, out_path, channels_to_merge=channels, nuclei_chan=args.nuclei_channels)