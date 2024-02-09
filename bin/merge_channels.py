#!/usr/bin/env python

"""
Script for merging channels into one. 
It will be used for an approximation of a cytoplasm channel, as our channels are only specific to a cell subpopulation.
"""
from pandas import read_csv
import numpy as np
import ome_types
import tifffile
import argparse
import os
from scipy.ndimage import gaussian_filter

from utils import read_tiff_orion, _tile_generator

def compute_hist(img, channel, x, y, chunk_x, chunk_y, img_min=None, img_max=None, num_bin=100, max_bin=0.9):
    """
    Compute the histogram of a channel from an image and get automatically min and max index for normalizing image afterward.
    The method used to get those are more or less the one used to get it manually. 
    'a little' after the pic (background) and remove 10% extremum for max
    
    Parameters
    ----------

    img: np.array
        image to analyze
    channel: int
        index of the channel of interest
    x: int
        height of img
    y: int
        witdh of img
    chunk_x: int
        the size in x of the tile to compute hist from
    chunk_y: int
        the size in y of the tile to compute hist from
    img_min: int
        minimal intensity value from img (can not be computed easily without img in full memory)
    img_max: int
        maximal intensity value from img
    num_bin: int
        number of bin for histogram (more = more precise, less = efficient)    
    max_bin: float
        max percentage to keep

    Return
    ------

    tuple of int
        value of the bin to normalize img.
    """
    if img_min is None or img_max is None:
        img_min, img_max = 0, 65535
    bins = np.linspace(img_min, img_max, num_bin)
    hist = np.zeros(num_bin-1)
    for tile in _tile_generator(img, channel, x, y, chunk_x, chunk_y):
        hist += np.histogram(tile, bins)[0]

    idx_min = hist.argmax()+1
    idx_max = int(num_bin * max_bin)
    idx_max = idx_max if len(hist) > idx_max > idx_min else idx_min + 1

    if idx_min >= len(hist)-2:
        return bins[-2], bins[-1]
    else:
        return bins[idx_min], bins[idx_max]


def tile_generator(arr, nuclei_chan, to_merge_chan, x, y, chunk_x, chunk_y, agg=np.max, norm='hist', norm_val=None):
    """
    generate tile and compute the merge and normalization on the fly

    Parameters
    ----------

    arr: np.array
        image to get tile from
    nuclei_chan: int
        index of the channel for nuclei
    to_merge_chan: list of int
        indexes of the channels to be merged
    x: int
        height of arr
    y: int
        witdh of arr
    chunk_x: int
        height of the tile
    chunk_y: int
        witdh of the tile
    agg: callable
        function to aggregate channel from to_merge_chan
    norm: str
        name of the function to get values to normalize channels before merge
    norm_val: None or list of float
        if norm = "custom" it will be used to normalize data

    Yield
    -----

    tile of the nuclei channel untouched and tile merged and normalized for others

    """
    norm_min = np.zeros(shape=max(nuclei_chan, max(to_merge_chan))+1)
    norm_max = np.zeros(shape=max(nuclei_chan, max(to_merge_chan))+1)
    for ci in [nuclei_chan, to_merge_chan]:
        if norm == 'hist':
            # first pass for normalisation
            for c in (ci if not isinstance(ci, int) else [ci]):
                norm_min[c], norm_max[c] = compute_hist(arr, c, x, y, chunk_x, chunk_y, img_min=None, img_max=None)

        for tmp_arr in _tile_generator(arr, ci, x, y, chunk_x, chunk_y):
            if norm == 'hist':
                tmp_arr = gaussian_filter(tmp_arr, 0.2)
                # normalize tmp_arr based on fitted curve or hist props
                tmp_arr = (tmp_arr - norm_min[ci, None, None]) / (norm_max[ci, None, None] - norm_min[ci, None, None])
                tmp_arr[tmp_arr < 0] = 0
                tmp_arr[tmp_arr > 1] = 1
                tmp_arr = (tmp_arr * 65535)
            elif norm == "gaussian":
                tmp_arr = gaussian_filter(tmp_arr, 1)
            elif norm == "custom" and norm_val is not None and ci == to_merge_chan:
                tmp_arr = tmp_arr.astype('float')
                for i, c in enumerate(ci):
                    tmp_arr[i] = (tmp_arr[i] - norm_val[c][0]) / (norm_val[c][1] - norm_val[c][0])
                tmp_arr[tmp_arr < 0] = 0
                tmp_arr[tmp_arr > 1] = 1
                tmp_arr = (tmp_arr * 65535)                

            if ci != to_merge_chan:
                yield tmp_arr.astype('uint16')
            else:
                yield agg(tmp_arr, axis=0).astype('uint16')
    tmp_arr = None # don't wait for next iteration to flush this


def merge_channels(in_path, out_path, nuclei_chan=0, channels_to_merge=None, chunk_size=(256,256), agg=np.max, norm=None, norm_val=None):
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
    norm: str
        name of normalization for channels, accepts either 'hist' (filter pixel based on histogram intensity) 
        'gaussian' for gaussian filter, 'custom' to custom normalization based on norm_val. None for no normalization.
    norm_val: dict of list of int
        value to normalize each channel separately (only work if norm 'custom' is selected)

    Returns
    -------
    None

    """
    if nuclei_chan in channels_to_merge:
        raise ValueError("There is conflict between channels to merge and nuclei channels")
    img_level, metadata = read_tiff_orion(in_path)
    try:
        nuclei_chan_metadata = metadata.get_channel(nuclei_chan)
    except IndexError:
        nuclei_chan_metadata = None

    metadata.remove_all_channels()

    if nuclei_chan_metadata is not None:
        metadata.add_channel(nuclei_chan_metadata)
    else:
        metadata.add_channel_metadata(channel_name="nuclear_channel")
    metadata.add_channel_metadata(channel_name="merged_channels")
    metadata.dtype='uint16'

    # todo : add annotation about channels used

    if channels_to_merge is None:
        channels_to_merge = list(range(2, img_level.shape[0]))

    with tifffile.TiffWriter(out_path, bigtiff=True, shaped=False) as tiff_out:
            tiff_out.write(
                data=tile_generator(img_level, nuclei_chan, channels_to_merge, 
                                    *img_level.shape[1:], *chunk_size, agg=agg, norm=norm, norm_val=norm_val),
                shape=(2, *img_level.shape[1:3]),
                tile=chunk_size,
                **metadata.to_dict(shape=img_level.shape[1:3])
            )

def guess_channels_to_merge(img_path):
    """we want to keep the nuclei marker and 
    remove the autofluorescence channel before merging"""
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
    normalization_col_name = "normalization"
    marker_name = "marker_name"
    mrk = read_csv(markers_path)
    result = {}
    norm = {}
    if segmentation_col_name in mrk.columns:
        mrk[segmentation_col_name] = mrk[segmentation_col_name].fillna(False).replace(
            {'False': False, "false": False, "Faux": False, 
             "faux": False, 'non': False, "no": False}
        ).astype(bool)
    else:
        print(f"no column {segmentation_col_name} found in markers.csv... guessing channels")
        return guess_channels_to_merge(img_path)[1], None
    for idx, row in mrk.iterrows():
        if row[segmentation_col_name]:
            result[idx] = row[marker_name]
            if normalization_col_name in row:
                try:
                    norm[idx] = [int(k) for k in row[normalization_col_name].split(';')]
                except AttributeError:
                    norm[idx] = [0, 65535]
    # need to compare channel name with metadata to get order 
    # (or its the same in both and we dont care)
    return list(result.keys()), norm or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="Input Image Path")
    parser.add_argument('--out', type=str, required=False, help="Output merged image Path")
    parser.add_argument('--channels', type=str, required=False, default=None, 
                        help="comma separated list or file with channels index to be merged (by default use every channels except the first)")
    parser.add_argument("--nuclei-channels", type=str, required=False, default=0, help="index of nuclei channel")
    parser.add_argument("--norm", type=str, required=False, default=None, help="normalization channels type")
    args = parser.parse_args()

    norm_val = None
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
                channels, norm_val = parse_markers(in_path, channels)
                norm = "custom" if norm_val is not None else args.norm if args.norm is not None else "hist"
            else:
                raise ValueError('Wrong format for channels')
    else:
        _, channels = guess_channels_to_merge(in_path)
    merge_channels(in_path, out_path, channels_to_merge=channels, nuclei_chan=args.nuclei_channels, norm=args.norm, norm_val=norm_val)
 