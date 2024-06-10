#!/usr/bin/env python

import argparse

import numpy as np
from tifffile import TiffFile, imwrite
from cellpose.dynamics import steps2D_interp, get_masks, remove_bad_flow_masks
from cellpose.transforms import resize_image
from cv2 import INTER_NEAREST

from utils import OmeTifffile
from dask_utils import correct_edges_inplace, compute_current_cell_id

import fastremap
from scipy.ndimage import maximum_filter1d
import dask.array as da

# a lot of those function are from https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py
# but were adaptated to be used in a memory efficient way when dealing with large images


def get_masks(p, iscell=None, rpad=20, cell_id=0):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    
    Parameters
    ----------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k + cell_id

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
        M0 = fastremap.mask(M0, bigc)
    # fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0

def follow_flows(dP, niter=200, device=None, block_info=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)

    if block_info is None:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    else:
        xs, xf = block_info[0]['array-location'][1]
        ys, yf = block_info[0]['array-location'][2]
        p = np.meshgrid(np.arange(xs, xf), np.arange(ys, yf), indexing='ij')
    niter = np.uint32(niter)
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        print('WARNING: no mask pixels found')
        return p
    
    p_interp = steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, device=device)    
    p[:,inds[:,0],inds[:,1]] = p_interp
    return p

def compute_masks(flows, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, 
                   min_size=15, resize=None, 
                   use_gpu=False,device=None, block_info=None):
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

    device: str
        name of the device where the calculation happen
    """
    dP = flows[:-1]
    cellprob = flows[-1]
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            p = follow_flows(dP * cp_mask / 5., niter=niter, 
                                            use_gpu=use_gpu, device=device)
        
        current_cell_id = compute_current_cell_id(block_info)
        
        #calculate masks
        mask = get_masks(p, iscell=cp_mask, cell_id=current_cell_id)
            
        # flow thresholding factored out of get_masks
        if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold, device=device)
        
        if resize is not None:
            mask = mask.astype(np.float32)
            mask = resize_image(mask, resize[0], resize[1], interpolation=INTER_NEAREST)
            mask = mask.astype(np.uint32)

    else: # nothing to compute, just make it compatible
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask


    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    # mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
    return mask
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="filename of flows in npy format")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--chunks', type=int, nargs=2, required=False, default=(1024, 1024), help="Size of chunk for dask")
    parser.add_argument('--overlap', type=int, required=False, default=60, help="Overlap (in pixel) for dask to perform computing of masks on chunks")
    args = parser.parse_args()

    flows = np.lib.format.open_memmap(vars(args)['in'])
    flows_da = da.from_array(flows, chunks=[3, *args.chunks])

    masks_graph = da.map_overlap(compute_masks, flows_da, dtype=np.uint32, depth={0: 0, 1: args.overlap, 2: args.overlap}, drop_axis=0)
    mask_memmap = np.lib.format.open_memmap(".tmp_masks.npy", mode='w+', dtype=np.uint32, shape=flows.shape[1:])

    da.store(masks_graph, mask_memmap, compute=True)

    correct_edges_inplace(mask_memmap, chunks_size=args.chunks)
    fastremap.renumber(mask_memmap, in_place=True) #convenient to guarantee non-skipped labels

    metadata = OmeTifffile(TiffFile(args.original).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = mask_memmap.dtype

    imwrite(args.out, mask_memmap, bigtiff=True, shaped=False, **metadata.to_dict(shape=mask_memmap.shape))
