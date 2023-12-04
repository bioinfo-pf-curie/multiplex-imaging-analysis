#!/usr/bin/env python

import os
import argparse

import numpy as np
from tifffile import TiffFile, imwrite
from cellpose.dynamics import follow_flows, get_masks, remove_bad_flow_masks
from cellpose.utils import fill_holes_and_remove_small_masks
from cellpose.transforms import resize_image
from cv2 import INTER_NEAREST

from utils import OmeTifffile
import sys
import psutil


def compute_masks(dP, cellprob, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, interp=True, 
                   min_size=15, resize=None, 
                   use_gpu=False,device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary """
    print(f"dP : {sys.getsizeof(dP)}, cellprob : {sys.getsizeof(cellprob)}")
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            p, inds = follow_flows(dP * cp_mask / 5., niter=niter, interp=interp, 
                                            use_gpu=use_gpu, device=device)
            print(f"p : {sys.getsizeof(p)}, {type(p)}")
            if inds is None:
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p
        #calculate masks
        mask = get_masks(p, iscell=cp_mask)
        print(f"mask : {sys.getsizeof(mask)}, {type(mask)}")
            
        # flow thresholding factored out of get_masks
        shape0 = p.shape[1:]
        if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device)
        
        if resize is not None:
            #if verbose:
            #    dynamics_logger.info(f'resizing output with resize = {resize}')
            if mask.max() > 2**16-1:
                recast = True
                mask = mask.astype(np.float32)
            else:
                recast = False
                mask = mask.astype(np.uint16)
            mask = resize_image(mask, resize[0], resize[1], interpolation=INTER_NEAREST)
            if recast:
                mask = mask.astype(np.uint32)
            Ly,Lx = mask.shape
        elif mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p


    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)

    return mask, p


def get_weight(tile, edge=False):
    xm = np.arange(tile)
    mean = xm.mean()
    if edge == 'f':
        xm[xm < mean] = mean 
    elif edge == 'l':
        xm[xm > mean] = mean
    xm = np.abs(xm - mean)
    return 1/(1 + np.exp((xm - (tile/2-20)) / 7.5))
        

def sum_of_weight_on_axis(tile_height, overlap, img_height):
    per_title = get_weight(tile_height)
    result = np.zeros(img_height)
    for cur_height in range(0, img_height, int(tile_height * (1-overlap))):
        cur_length = min(len(per_title), img_height-cur_height)
        # zarr doesnt support incorrect length in indexing
        if not cur_height:
            added_weight = get_weight(tile_height, edge="f")
        elif cur_length < tile_height or cur_height == img_height:
            added_weight = get_weight(tile_height, edge='l')[:cur_length]
        else:
            added_weight = per_title
        result[cur_height:cur_height+cur_length] += added_weight
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
    total_flow = np.memmap('.tmp_flow.arr', dtype='float32', mode="write", shape=(3, *original_tiff.series[0].shape[1:]))
    for npy in list_npy:
        cur_height = get_current_height(npy)
        flow = load_npy(npy)
        weighted_flow = np.array(flow[4]) * get_weight(flow[4].shape[1])[np.newaxis, :, np.newaxis]
        total_flow[:, cur_height:cur_height+weighted_flow.shape[1], :] += weighted_flow
        total_flow.flush()
    flow = None # can be collected
    y_weight = sum_of_weight_on_axis(tile, overlap, original_tiff.series[0].shape[1])
    for chunk in range(0, total_flow.shape[2], tile):
        total_flow[..., chunk:chunk+tile] = total_flow[..., chunk:chunk+tile] / y_weight
        total_flow.flush()
    return total_flow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    args = parser.parse_args()

    flows = stich_flow(vars(args)['in'], args.original)
    masks = compute_masks(flows[:-1], flows[-1])[0] # todo : modifier ça pour ne pas tout mettre en mémoire

    metadata = OmeTifffile(TiffFile(args.original).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")
    metadata.dtype = masks.dtype

    imwrite(args.out, masks, bigtiff=True, shaped=False, **metadata.to_dict())
