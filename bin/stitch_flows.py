#!/usr/bin/env python
from utils import get_current_height

import argparse
import numpy as np
from tifffile import TiffFile

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
    return np.ascontiguousarray(result[:, np.newaxis])


def load_npy(npy_path):
    """Helper to load npy files"""
    return np.load(npy_path, allow_pickle=True).item()['flows']

def stich_flow(list_npy, input_img_path, overlap, out_path):
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
    flow_shape = (3, *original_tiff.series[0].shape[1:])
    # init memmap
    total_flow = np.lib.format.open_memmap(out_path, dtype='float32', mode="w+", shape=flow_shape)

    tiles_height = []
    
    for i, npy in enumerate(list_npy):
        cur_height = get_current_height(npy)
        flow = load_npy(npy)
        weight = get_weight(flow[4].shape[1], edge=("f" if not i else "l" if i == len(list_npy) - 1 else None))
        weighted_flow = np.ascontiguousarray(np.array(flow[4]) * weight[np.newaxis, :, np.newaxis]) # accelerate writing operation
        tiles_height.append(weighted_flow.shape[1])
        total_flow[:, cur_height:cur_height+weighted_flow.shape[1], :] += weighted_flow
        if not i % 10: # flush every ten file (~10GB)
            total_flow.flush()
            # reload memmap each time else it will accumulate in memory
            total_flow = np.lib.format.open_memmap(out_path, dtype='float32', shape=flow_shape)
    total_flow.flush()
    total_flow = np.lib.format.open_memmap(out_path, dtype='float32', shape=flow_shape)
    del flow # can be collected
    tile_height = int(np.median(tiles_height)) # last one may be cut
    y_weight = sum_of_weight_on_axis(tile_height, overlap, original_tiff.series[0].shape[1])
    chunk_count = 0

    for chunk in range(0, flow_shape[2], tile_height):
        total_flow[..., chunk:chunk+tile_height] /= y_weight
        chunk_count += 1 
        if not chunk_count % 10:
            total_flow.flush()
            total_flow = np.lib.format.open_memmap(out_path, dtype='float32', shape=flow_shape)
    total_flow.flush() # update last chunk
    del total_flow # no close method on memmap
    return

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
        np.save(args.out, flows)
    else:
        stich_flow(list_npy, args.original, overlap=args.overlap, out_path=args.out)
