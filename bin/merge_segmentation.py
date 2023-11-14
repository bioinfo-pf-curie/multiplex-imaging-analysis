#!/usr/bin/env python

import os
import argparse

import numpy as np
from tifffile import TiffFile, imwrite
from cellpose.dynamics import compute_masks

from utils import transfer_metadata


def get_weight(tile):
    xm = np.arange(tile)
    xm = np.abs(xm - xm.mean())
    return 1/(1 + np.exp((xm - (tile/2-20)) / 7.5))
        

def sum_of_weight_on_axis(tile_height, overlap, img_height):
    per_title = get_weight(tile_height)
    result = np.zeros(img_height)
    for cur_height in range(0, img_height, int(tile_height * (1-overlap))):
        cur_length = min(len(per_title), img_height-cur_height)
        result[cur_height:cur_height+cur_length] += per_title[:cur_length]
    return result[:, np.newaxis]


def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=True).item()['flows']

def get_current_height(npy_path):
    npy_name = os.path.basename(npy_path)
    while True:
        npy_name, height = npy_name.rsplit('_', 1)
        try:
            return int(height)
        except ValueError:
            pass

def stich_flow(list_npy, input_img_path, tile=224, overlap=.1):
    original_tiff = TiffFile(input_img_path)
    total_flow = np.zeros((3, *original_tiff.series[0].shape[1:]))
    for npy in list_npy:
        cur_height = get_current_height(npy)
        flow = load_npy(npy)
        weighted_flow = np.array(flow[4]) * get_weight(flow[4].shape[1])[np.newaxis, :, np.newaxis]
        total_flow[:, cur_height:cur_height+weighted_flow.shape[1], :] += weighted_flow
    total_flow /= sum_of_weight_on_axis(tile, overlap, original_tiff.series[0].shape[1])
    return total_flow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    args = parser.parse_args()

    flows = stich_flow(vars(args)['in'], args.original)
    masks = compute_masks(flows[:-1], flows[-1])[0] # todo : modifier ça pour ne pas tout mettre en mémoire
    imwrite(args.out, masks, ome=True, bigtiff=True, **transfer_metadata(TiffFile(args.original).pages[0], func='write'))
