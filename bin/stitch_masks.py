#!/usr/bin/env python

from utils import get_current_height
import argparse

import numpy as np
import tifffile 
from scipy.signal import windows


def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (windows.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (windows.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window



def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(window_x, -1)
    window_y = np.expand_dims(window_y, -1)

    window = window_x * window_y.transpose(1, 0)
    return window




def stich_masks(list_mask_chunks, input_img_path, overlap, out_path):
    """
    Merge a list of masks (in tiff format) into a mask for the complete image
    
    Parameters
    ----------
    
    list_mask_chunks: list of path or str
        list of tiff file to get masks from
        
    input_img_path: path or str
        path of the original image (get metadata from)
        
    overlap: float
        the overlap used between tiles
        
    Return
    ------
    
    result: np.array
        the mask for complete image
    """
    min_tile_size = 32

    original_tiff = tifffile.TiffFile(input_img_path)
    original_shape = original_tiff.series[0].shape[1:]
    result = tifffile.memmap(out_path, dtype=float, shape=original_shape)

    # previous_cells = None
    for chunk in list_mask_chunks:
        cur_height = get_current_height(chunk)
        img = tifffile.imread(chunk)
        w = window_2D(img.shape, overlap_x=[int(img.shape[0] * overlap)] * 2, overlap_y=(0, 0))
        if (min_tile_size <= img.shape[0] < original_shape[0]):
            result[cur_height:cur_height+img.shape[0], :] += img * w
        else:
            result[cur_height:cur_height+img.shape[0], :] = img
        result.flush()

    img = None # can be collected
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, nargs='+', help="list of Image Path (cropped) to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="value of overlap used for splitting images")
    args = parser.parse_args()

    list_chunks = vars(args)['in']
    stich_masks(list_chunks, args.original, args.overlap, args.out)
