#!/usr/bin/env python

import argparse

import numpy as np
from tifffile import TiffFile, imwrite
from cellpose.dynamics import steps2D_interp, get_masks, remove_bad_flow_masks, map_coordinates
from cellpose.utils import fill_holes_and_remove_small_masks
from cellpose.transforms import resize_image
from cv2 import INTER_NEAREST

from utils import OmeTifffile
import sys

import plotly.graph_objects as go
import fastremap
from scipy.ndimage import maximum_filter1d
import dask.array as da


def get_masks(p, iscell=None, rpad=20, cell_id=0):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    Parameters
    ----------------
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

    # everything before this should be chunked (should be possible to)
    # make M a memmap

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

def follow_flows(dP, mask=None, niter=200, use_gpu=True, device=None, block_info=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    
    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


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
        # print(block_info[0]['shape'])
        xs, xf = block_info[0]['array-location'][1]
        # xs -= 60
        # xf += 60
        ys, yf = block_info[0]['array-location'][2]
        # ys -= 60
        # yf += 60
        p = np.meshgrid(np.arange(xs, xf), np.arange(ys, yf), indexing='ij')
    niter = np.uint32(niter)
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        print('WARNING: no mask pixels found')
        return p
    
    # p_interp = steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, use_gpu=use_gpu, device=device)    
    p_interp = p[:,inds[:,0], inds[:,1]]
    dPt = np.zeros(p_interp.shape, np.float32)
            
    for t in range(niter):
        # if not t:
        #     print(dP.astype(np.float32).shape)
        #     print(p_interp[0].shape)
        #     print(p_interp[1])

            # (2, 1024, 1024)
            # (304495,)
            # [176. 177. 178. ... 778. 779. 780.]

            # (2, 256, 256)
            # (22372,)
            # [787. 788. 789. ... 942. 943. 944.]

        map_coordinates(dP.astype(np.float32), p_interp[0], p_interp[1], dPt)
        for k in range(len(p_interp)):
            p_interp[k] = np.minimum(shape[k]-1, np.maximum(0, p_interp[k] + dPt[k]))

    p[:,inds[:,0],inds[:,1]] = p_interp
    return p

def compute_masks(flows, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, interp=True, 
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
*block_info[0]['array-location'][1]

    device: str
        name of the device where the calculation happen
    """
    dP = flows[:-1]
    cellprob = flows[-1]
    # dP = da.from_array(flows[:-1], chunks=[2, 256, 256])
    # cellprob = da.from_array(flows[-1], chunks=[256, 256])
    # print(f"dP : {sys.getsizeof(dP)}, cellprob : {sys.getsizeof(cellprob)}")
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            # dP_da = da.from_array(dP * cp_mask / 5., chunks=[2, 256, 256])
            # test = dP * cp_mask / 5.
            # p = da.map_overlap(follow_flows, test,# depth={0: 0, 1: 60, 2: 60}, 
            #                          niter=niter, use_gpu=use_gpu, device=device)
            # # to_zarr(".tmp_p.zarr", object_codec=numcodecs.JSON(), compute=True, return_stored=True)
            p = follow_flows(dP * cp_mask / 5., niter=niter, 
                                            use_gpu=use_gpu, device=device)
            # print(f"p : {sys.getsizeof(p)}, {type(p)}")
            # if inds is None:
            #     shape = resize if resize is not None else cellprob.shape
            #     mask = np.zeros(shape, np.uint16)
            #     p = np.zeros((len(shape), *shape), np.uint16)
            #     return mask, p
        #calculate masks
        current_chunk = block_info[0]['chunk-location']
        total_chunk = block_info[0]['num-chunks']
        current_cell_id = int((current_chunk[1] +  current_chunk[2] * total_chunk[1]) * np.multiply(*dP.shape[1:]) / 900)
        # mean cell area (determine by cellpose parameters)
        mask = get_masks(p, iscell=cp_mask, cell_id=current_cell_id)
        # p = None
        # mask = p#.compute()

        # go.Figure(go.Heatmap(z=mask[0]), layout={'title_text': "mask"}).show('browser')
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
    # mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
    # print(f"{mask=}")
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="filename of flows in npy format")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--overlap', type=float, required=False, default=0.1, help="value of overlap used for splitting images")
    args = parser.parse_args()
    flows = np.lib.format.open_memmap(vars(args)['in'])
    # masks = compute_masks(flows)[0] # todo : modifier ça pour ne pas tout mettre en mémoire
    # replacer 60 par 2 * la taille d'une cellule (pour etre sur de tout avoir)

    flows_da = da.from_array(flows, chunks=[3, 256, 256])
    masks = da.map_overlap(compute_masks, flows_da, dtype=np.uint16, depth={0: 0, 1: 60, 2: 60}, drop_axis=0).compute()
    remap_chunked = {}
    for limit_x in range(256, masks.shape[0], 256):
        chunk1 = masks[limit_x-1]
        chunk2 = masks[limit_x]
        # limit_arr = masks[limit_x-1:limit_x+1]
        replace = {}
        for k, v in zip(chunk1[chunk1.astype(bool)], chunk2[chunk1.astype(bool)]):
            if k not in replace:
                replace[k] = [(v, 1)]
            else:
                if not v:
                    continue
                elif v == replace[k][0]:
                    replace[k][1] += 1
                else:
        print(replace)
        # faire la somme des deux lignes 
        # créer un dict ancienne valeur: somme (modifié pour etre unique dans le mask final)
        # si l'ancienne valeur existe deja comparé le nombre de pixel et prendre le majoritaire
        break
    # masks_da, _ = ndmeasure.label(masks_da)
    # masks_comparison([masks, masks_da], names=['normal', 'dask'])
    # fastremap.renumber(masks, in_place=True) #convenient to guarantee non-skipped labels
    go.Figure(go.Heatmap(z=masks), layout={'title_text': "masks_da"}).show('browser')
    # metadata = OmeTifffile(TiffFile(args.original).pages[0])
    # metadata.remove_all_channels()
    # metadata.add_channel_metadata(channel_name="masks")
    # metadata.dtype = masks.dtype

    # imwrite(args.out, masks, bigtiff=True, shaped=False, **metadata.to_dict())
