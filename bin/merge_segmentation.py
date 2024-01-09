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

import plotly.express as px
import plotly.graph_objects as go
import numcodecs

def compute_masks(flows, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, interp=True, 
                   min_size=15, resize=None, 
                   use_gpu=False,device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary 
    (from https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py#L740)
    
    Parameters
    ----------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    cellprob: 
        cell probability

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    niter: int
        number of iteration in follow_flows
    
    cellprob_threshold: float
        threshold for cell probability 

    flow_threshold: float
        threshold for flow probability

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    min_size: int
        minimal size for a mask to be kept

    resize: list of int or None
        if set, mask will be resized to this size

    use_gpu: bool
        flag to use gpu or not (default False)

    device: str
        name of the device where the calculation happen
    """
    dP = flows[:-1]
    cellprob = flows[-1]
    # print(f"dP : {sys.getsizeof(dP)}, cellprob : {sys.getsizeof(cellprob)}")
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            dP_da = da.from_array(dP * cp_mask / 5., chunks=[2, 256, 256])
            p = da.map_overlap(follow_flows, dP_da, depth={0: 0, 1: 60, 2: 60}, 
                                     niter=niter, interp=interp, use_gpu=use_gpu, device=device).to_zarr(".tmp_p.zarr", object_codec=numcodecs.JSON(), compute=True, return_stored=True)
            # p, inds = follow_flows(dP * cp_mask / 5., niter=niter, interp=interp, 
            #                                 use_gpu=use_gpu, device=device)
            print(f"p : {sys.getsizeof(p)}, {type(p)}")
            # if inds is None:
            #     shape = resize if resize is not None else cellprob.shape
            #     mask = np.zeros(shape, np.uint16)
            #     p = np.zeros((len(shape), *shape), np.uint16)
            #     return mask, p
        #calculate masks

        mask = get_masks(p, iscell=cp_mask)
        p = None
        # print(f"mask : {sys.getsizeof(mask)}, {type(mask)}")
            
        # flow thresholding factored out of get_masks
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

        elif mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask


    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
    # print(f"{mask=}")
    return mask[None, ...]


def get_weight(tile, edge=False):
    """
    Compute weight of tile along height (on the edge of a tile the weight will be less 
    except if it is also the edge of final image)
    
    Parameters
    ----------
    
    tile: np.array
        tile considered
    edge: bool or str
        if false the weight of the tile is less on the edges of the height

    Return
    ------

    a vector of weight along the height of the tile.
    """
    if edge == "both":
        return np.ones(tile)
    xm = np.arange(tile)
    mean = xm.mean()
    if edge == 'f':
        xm[xm < mean] = mean 
    elif edge == 'l':
        xm[xm > mean] = mean
    xm = np.abs(xm - mean)
    return 1/(1 + np.exp((xm - (tile/2-20)) / 7.5))
        

def sum_of_weight_on_axis(tile_height, overlap, img_height):
    """
    Compute the total weight along the height of final image
    
    Parameters
    ----------
    
    tile_height: int
        height of individual tile
        
    overlap: float
        percentage of overlap between tiles
        
    img_height: int
        total height of the image
        
    Return
    ------
    
    A vector of weight along total height
    """
    per_title = get_weight(tile_height)
    result = np.zeros(img_height)
    for cur_height in range(0, img_height, int(tile_height * (1-overlap))):
        cur_length = min(len(per_title), img_height-cur_height)
        # zarr doesnt support incorrect length in indexing
        if not cur_height:
            added_weight = get_weight(tile_height, edge="f")
        elif cur_height + tile_height >= img_height:
            added_weight = get_weight(tile_height, edge='l')[:cur_length]
        else:
            added_weight = per_title
        result[cur_height:cur_height+cur_length] += added_weight
    return result[:, np.newaxis]


def load_npy(npy_path):
    """Helper to load npy files"""
    return np.load(npy_path, allow_pickle=True).item()['flows']

def get_current_height(npy_path):
    """Helper to parse filename to get position in height for the corresponding tile"""
    npy_name = os.path.basename(npy_path)
    while True:
        npy_name, height = npy_name.rsplit('_', 1)
        try:
            return int(height)
        except ValueError:
            pass
        if not npy_name:
            raise ValueError(f'Height of image {npy_name} not found')

def stich_flow(list_npy, input_img_path, overlap):
    """
    Merge a list of flows (in npy format) into a flows for the complete image
    
    Parameters
    ----------
    
    list_npy: list of path or str
        list of npy file to get flow from
        
    input_img_path: path or str
        path of the original image (get metadata from)
        
    overlap: float
        the overlap used between tiles
        
    Return
    ------
    
    total_flow: np.array
        the flow for complete image
    """
    original_tiff = TiffFile(input_img_path)
    total_flow = np.memmap('.tmp_flow.arr', dtype='float32', mode="write", shape=(3, *original_tiff.series[0].shape[1:]))
    tile_height = None

    for i, npy in enumerate(list_npy):
        cur_height = get_current_height(npy)
        flow = load_npy(npy)
        weight = get_weight(flow[4].shape[1], edge=("f" if not i else "l" if i == len(list_npy) - 1 else None))

        weighted_flow = np.array(flow[4]) * weight[np.newaxis, :, np.newaxis]
        if tile_height is None:
            tile_height = weighted_flow.shape[1]
        total_flow[:, cur_height:cur_height+weighted_flow.shape[1], :] += weighted_flow
        total_flow.flush()
    flow = None # can be collected
    y_weight = sum_of_weight_on_axis(tile_height, overlap, original_tiff.series[0].shape[1])
    for chunk in range(0, total_flow.shape[2], tile_height):
        total_flow[..., chunk:chunk+tile_height] = total_flow[..., chunk:chunk+tile_height] / y_weight
        total_flow.flush()
    return total_flow

# @profile
def empty():
    print('its just to see')


color = px.colors.qualitative.Plotly
color[0] = "black"
color[7] = "white"

def binarize(img):
    res = (img - np.min(img)) / (np.max(img) - np.min(img))
    res[res < .5] = 0
    res[res >= .5] = 1
    return res


def compare(outline1, outline2):
    return 100 * np.sum(binarize(outline1) - binarize(outline2)) / np.prod(outline1.shape) 

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        if j: y += int(j)<<i
    return y


def masks_comparison(list_of_masks, names=[]):
    result = np.apply_along_axis(bool2int, 0, np.stack(list_of_masks).astype(bool))
    if not names:
        names = range(len(list_of_masks))
    if len(list_of_masks) == 3:
        names = {0: 'pas de masque', 1: names[0], 2: names[1], 
                3: f'{names[0]} & {names[1]}', 4: names[2], 
                5: f'{names[0]} & {names[2]}', 6: f'{names[1]} & {names[2]}',
                7: "commun"}
        colorscale = [[0, 'black'], [0.05, 'black'], 
                      [0.05, '#EF553B'], [0.22, '#EF553B'], 
                      [0.22, '#00CC96'], [0.35, '#00CC96'], 
                      [0.35, '#AB63FA'], [0.5, '#AB63FA'], 
                      [0.5, '#FFA15A'], [0.63, '#FFA15A'], 
                      [0.63, '#19D3F3'], [0.77, '#19D3F3'], 
                      [0.77, '#FF6692'], [0.95, '#FF6692'], 
                      [0.95, 'white'], [1, 'white']]
    elif len(list_of_masks) == 2:
        names = {0: 'pas de masque', 1: names[0], 2: names[1], 3: "commun"}
        colorscale = [[0, 'black'], [0.1, 'black'], 
                      [0.1, '#EF553B'], [0.6, '#EF553B'], 
                      [0.6, '#00CC96'], [0.9, '#00CC96'], 
                      [0.9, 'white'], [1, 'white']]
    else:
        colorscale="sunset"
    fig = go.Figure(go.Heatmap(z=result[::-1, :], colorscale=colorscale, colorbar=dict(tickvals=list(names.keys()), ticktext=list(names.values()))))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show('browser')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="value of overlap used for splitting images")
    args = parser.parse_args()

    list_npy = vars(args)['in']
    if len(list_npy) == 1:
        flows = load_npy(list_npy[0])[4]
    else:
        flows = stich_flow(list_npy, args.original, overlap=args.overlap)

    import dask.array as da
    from dask_image import ndmeasure
    print(type(flows))

    masks = compute_masks(flows)[0] # todo : modifier ça pour ne pas tout mettre en mémoire
    # replacer 60 par 2 * la taille d'une cellule (pour etre sur de tout avoir)

    # flows_da = da.from_array(flows, chunks=[3, 256, 256])
    # masks_da = da.map_overlap(compute_masks, flows_da, dtype=np.uint16, depth={0: 0, 1: 60, 2: 60}).compute()[0].astype(bool)
    # masks_da, _ = ndmeasure.label(masks_da)
    # masks_comparison([masks, masks_da], names=['normal', 'dask'])

    metadata = OmeTifffile(TiffFile(args.original).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")
    metadata.dtype = masks.dtype

    imwrite(args.out, masks, bigtiff=True, shaped=False, **metadata.to_dict())
    # metadata.dtype = masks_da.dtype
    # imwrite(args.out, masks_da, bigtiff=True, shaped=False, **metadata.to_dict())
