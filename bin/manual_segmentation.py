#!/usr/bin/env python

# ==================== #
#       MODULES        #
# ==================== #

import os
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geojson
from collections import namedtuple
import io
import fastremap
import random
from skimage.draw import polygon, polygon_perimeter
from skimage.io import imsave
import tifffile


def get_outline_image(gj, height, width):
    n = len(gj)
    best_dtype = np.min_scalar_type(n)
    img = np.zeros((height, width), dtype=np.double) # heigth x width
    for idx, roi in enumerate(gj):
        if (idx in [0, 1]):
            pass
            # print(roi)
        try:
            poly = np.array(roi["geometry"]["coordinates"])
            a, b, c = poly.shape
            poly = np.reshape(poly, (b, c))
            rr, cc = polygon_perimeter(poly[:, 0], poly[:, 1])
            img[cc, rr] = 255
        except:
            pass

    return(img)

def get_mask_image(gj, height, width):
    n = len(gj)
    best_dtype = np.min_scalar_type(n)
    img = np.zeros((height, width), dtype=best_dtype) # heigth x width
    for idx, roi in enumerate(gj['features']):
        try:
            poly = np.array(roi["geometry"]["coordinates"])
            a, b, c = poly.shape
            poly = np.reshape(poly, (b, c))
            rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
            img[cc, rr] = idx
        except:
           pass

    return(img)

def g2o(args):
    gj_path = pathlib.Path(args.gjfile).expanduser()
    with open(gj_path, 'r') as gjfile:
        gj = geojson.load(gjfile)
    img = get_outline_image(gj, args.height, args.width)
    if args.out is None:
        mask_file_name = gj_path.stem.replace(".geojson", "_cellBoundaries.tif")
    else: 
        mask_file_name = args.out
    imsave(mask_file_name, img, plugin="tifffile")
    return mask_file_name

def g2m(args):
    gj_path = pathlib.Path(args.gjfile).expanduser()
    with open(gj_path, 'r') as gjfile:
        gj = geojson.load(gjfile)
    img = get_mask_image(gj, args.height, args.width)
    if args.out is None:
        mask_file_name = gj_path.stem.replace("_cellBoundaries", "").replace(".geojson", "_mask.tif")
    else: 
        mask_file_name = args.out
    imsave(mask_file_name, img, plugin="tifffile")

def make_tile(args):
    from utils import read_tiff_orion
    img_path = pathlib.Path(args.image).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()
    
    
    
    img, metadata = read_tiff_orion(img_path)
    position = [(random.choice(range(img.size[0] - args.size)), random.choice(range(img.size[1] - args.size))) for _ in range(args.nb)] if args.position == "random" else args.position
    for tile_pos in position:
        tile_pos = [int(t) for t in tile_pos.split(',')]
        out_name = out_dir / f"{img_path.stem}_tiled_{tile_pos[0]}_{tile_pos[1]}.tiff"
        with tifffile.TiffWriter(out_name, bigtiff=True, shaped=False) as tiff_out:
            if img.ndim == 3:
                tmp_arr = img[:, tile_pos[0]:tile_pos[0] + args.size, tile_pos[1]:tile_pos[1] + args.size]
            elif img.ndim == 2:
                tmp_arr = img[tile_pos[0]:tile_pos[0] + args.size, tile_pos[1]:tile_pos[1] + args.size]
            metadata.update_shape(tmp_arr.shape)
            tiff_out.write(data=tmp_arr, shape=tmp_arr.shape, **metadata.to_dict())

def determine_size(list_img):
    i = 0
    while list_img[i].endswith('.geojson'):
        i += 1
    return tifffile.TiffFile(list_img[i]).pages[0].shape

def compare_geojson(args):
    pass
    # shapely.geometry.shape(geojson)


def compare(args):
    gt = args.ground_truth
    if all(img.endswith(".geojson") for img in args.images) and gt.endswith(".geojson"):
        return compare_geojson(args)
    
    height, width = determine_size(args.images + [gt])
    g2m_args = namedtuple('args', ['gjfile', 'height', 'width', 'out'])

    if gt.endswith('.geojson'):
        tmp = io.BytesIO()
        g2m(g2m_args(gt, height, width, tmp))
        tmp.seek(0)
        gt = tmp

    gt = tifffile.imread(gt)
    gt_cells_nb = gt.max()

    for img_name in args.images:

        if img_name.endswith('.geojson'):
            tmp = io.BytesIO()
            g2m(g2m_args(img, height, width, tmp))
            tmp.seek(0)
            img = tifffile.imread(tmp)
        else:
            img = tifffile.imread(img_name)

        fastremap.renumber(img, in_place=True)

        bgt = gt > 0
        bimg = img > 0

        # pixel classification
        total = np.multiply(*bgt.shape)
        tp = np.sum(bgt & bimg)
        fp = np.sum(bimg & ~bgt)
        fn = np.sum(~bimg & bgt)
        ap = tp / (tp+fp+fn)
        print(f"{pathlib.Path(img_name).stem}\n")
        print(f"\tavg prec = {ap:.4f}\n")

        nb_cell = img.max()

        print(f"\tcell number = {nb_cell} ({gt_cells_nb} in gt)")
        iou_mean = {}
               
        # cell to cell comparison
        for label in range(1, nb_cell+1):
            label_mask = img == label
            gt_equivalent = gt[label_mask]
            u, c = np.unique(gt_equivalent[gt_equivalent!=0], return_counts=True)
            try:
                corresponding_label = u[c.argmax()]
            except ValueError:
                # no corresponding
                # iou_mean.append(0)
                continue
            gt_mask = gt == corresponding_label
            iou = (label_mask & gt_mask).sum() / (label_mask | gt_mask).sum()
            if corresponding_label in iou_mean and iou < iou_mean[corresponding_label]:
                continue
            iou_mean[corresponding_label] = iou
            print(f'\tcell {label} : {iou=:.02f}\n')
        print(f"\tiou mean = {sum(iou_mean.values()) / len(iou_mean)}\n")


# ==========
# Arguments
# ==========
def parse_args(args=None):
    parser = argparse.ArgumentParser(prog="Manual Segmentation")
    subparsers = parser.add_subparsers(help='sub-command help', required=True)
    parser_g2m = subparsers.add_parser('g2m')
    parser_g2m.add_argument('--gjfile', type=str,
                            help='the geojson file to convert into labelled masks.')
    parser_g2m.add_argument('--height', type=int,
                            help='The image height in pixels')
    parser_g2m.add_argument('--width', type=int,
                            help='The image width in pixels')
    parser_g2m.add_argument('-o', '--out', type=str,
                            default=sys.stdout,
                            help='Output file name. Defaults to stdout.')
    parser_g2m.set_defaults(func=g2m)

    parser_g2o = subparsers.add_parser('g2o')
    parser_g2o.add_argument('--gjfile', type=str,
                            help='the geojson file to convert into labelled masks.')
    parser_g2o.add_argument('--height', type=int,
                            help='The image height in pixels')
    parser_g2o.add_argument('--width', type=int,
                            help='The image width in pixels')
    parser_g2o.add_argument('-o', '--out', type=str,
                            default=sys.stdout,
                            help='Output file name. Defaults to stdout.')
    parser_g2o.set_defaults(func=g2o)
    
    parser_tile = subparsers.add_parser('tile')
    parser_tile.add_argument('--image', type=str,
                             help='Image to take tile from')
    parser_tile.add_argument('--size', type=int,
                             help='tile size will be (size,  size)')
    parser_tile.add_argument('--position', type=str, nargs="+",
                             help='either "random" or a list of position to take tile from')
    parser_tile.add_argument('--nb', type=int, help="If position is random, how many tile will be create (otherwise it is len(position))")
    parser_tile.add_argument('--out_dir', type=str,
                             help='Output directory name. if not exist, will be created')
    parser_tile.set_defaults(func=make_tile)
    
    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument('--ground_truth', type=str,
                             help='Ground truth file (geojson or mask)')
    parser_compare.add_argument('--images', type=str, nargs="+",
                             help='list of image (or geojson) to compare to gt')
    parser_compare.set_defaults(func=compare)
    
    return parser.parse_args(args)

# ==================== #
#         MAIN         #
# ==================== #
def main(args=None):
    args = parse_args(args)
    args.func(args)



if __name__ == '__main__':
    sys.exit(main())