#!/usr/bin/env python

# ==================== #
#       MODULES        #
# ==================== #

import sys
import pathlib
import argparse
import numpy as np
import geojson
from collections import namedtuple
import io
import fastremap
import random
from skimage.draw import polygon, polygon_perimeter
from skimage.io import imsave
import tifffile
from shapely import geometry, STRtree, GeometryCollection

from mask2geojson import mask2geojson

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
    position = [(random.choice(range(img.size[0] - args.size)), 
                 random.choice(range(img.size[1] - args.size))) 
                 for _ in range(args.nb)] if args.position == "random" else args.position
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

def geojson2shapely(geojson):
    result = []
    if isinstance(geojson, dict):
        geojson = geojson['features']
    if isinstance(geojson, list):
        for cell in geojson:
            result += [geometry.shape(cell.get('geometry', cell))]
    else:
        raise ValueError("Unkwown type for geojson file")
    return result

def compare(args):
    gt = args.ground_truth

    if not gt.endswith('.geojson'):
        gjson = mask2geojson(mask=tifffile.imread(gt))
    else:
        with open(gt, "r") as gtf:
            gjson = geojson.load(gtf)
    gt_cells = geojson2shapely(gjson)
    gt_tree = STRtree(gt_cells)
    gt_cells_nb = len(gt_cells)

    total = GeometryCollection(gt_cells).bounds
    total = total[2] * total[3]

    for gj_files in args.images:
        if not gj_files.endswith('.geojson'):
            gjson = mask2geojson(mask=tifffile.imread(gj_files))
        else:
            with open(gj_files, 'r') as gjf:
                gjson = geojson.load(gjf)
        other_cells = geojson2shapely(gjson)
        nb_cell = len(other_cells)
        common = gt_tree.query(other_cells, predicate="intersects")
        if common.size == 0:
            print('no common cells')
            return
        # filter non unique pair of common cells based on best intersection
        def intersect_area(cpl):
            return gt_cells[cpl[1]].intersection(other_cells[cpl[0]]).area

        common = np.array([
            max(common[:,common[0] == idx].T, key=intersect_area)
            for idx in np.unique(common[0])
        ]).T

        common = np.array([
            max(common[:,common[1] == idx].T, key=intersect_area)
            for idx in np.unique(common[1])
        ]).T

        ap_common = []
        iou_mean = []

        not_found = [gt_cells[i] for i in range(len(gt_cells)) if i not in common[1]]
        not_cells = [other_cells[i] for i in range(len(other_cells)) if i not in common[0]]

        tp = 0
        fp = sum([c.area for c in not_cells])
        fn = sum([c.area for c in not_found])

        tpcp = 0
        fpcp = len(not_cells)
        fncp = len(not_found)
        
        for paired_cells in common.T:
            gtc = gt_cells[paired_cells[1]]
            oc = other_cells[paired_cells[0]]

            intersect = gtc.intersection(oc).area
            too_much = oc.difference(gtc).area
            not_enough = gtc.difference(oc).area

            tp += intersect
            fp += too_much
            fn += not_enough

            iou = intersect / gtc.union(oc).area
            ap = intersect / (intersect + too_much + not_enough)

            if iou > 0.5:
                tpcp += 1
            else:
                fpcp += 1
                fncp += 1
            ap_common.append(ap)
            iou_mean.append(iou)

            
        print(f"{pathlib.Path(gj_files).stem}\n")
        print(f"\tfound {nb_cell} (with {len(not_cells)} false cells and {len(not_found)} cells not found) cells out of {gt_cells_nb} in ground truth\n")
        if len(ap_common):
            print(f"\tavg prec = {sum(ap_common) / len(ap_common):.4f}\n")
        if (tpcp + fpcp + fncp):
            ap_cellpose = tpcp / (tpcp + fpcp + fncp)
            print(f"\tcellpose avg prec = {ap_cellpose:.4f}\n")
        print(f"\tiou mean = {sum(iou_mean) / len(iou_mean)}\n")
        tn = (total - (tp + fp + fn)) / total
        tp /= total
        fp /= total
        fn /= total
        print(f"\ttotal pixel classification = \n")
        print("\t+--------+------+------+")
        print("\t|        |  POS |  NEG |")
        print(f"\t| TRUE   | {tp:.02f} | {tn:.02f} |")
        print(f"\t| FALSE  | {fp:.02f} | {fn:.02f} |")
        print("\t+--------+------+------+")
        print(f'\n\t F1 score = {2*tp / (2*tp + fp + fn):.02f}\n\n')
            

def compare_img(args):
    gt = args.ground_truth
    if all(img.endswith(".geojson") for img in args.images) and gt.endswith(".geojson"):
        return compare(args)
    
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

def m2g(args):
    gjson = mask2geojson(mask=tifffile.imread(args.mask), object_type=args.object_type, 
                           connectivity=args.connectivity, transform=args.transform,
                           downsample=args.downsample, include_labels=args.include_labels,
                           classification=args.classification)
    if args.out is None:
        args.out = pathlib.Path(args.mask).stem + ".geojson"
    with open(args.out, "w") as out:
       geojson.dump(gjson, out)
    

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
    parser_tile.add_argument('--out_dir', type=str, default=".",
                             help='Output directory name. if not exist, will be created')
    parser_tile.set_defaults(func=make_tile)
    
    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument('-gt', '--ground_truth', type=str,
                             help='Ground truth file (geojson or mask)')
    parser_compare.add_argument('--images', type=str, nargs="+",
                             help='list of image (or geojson) to compare to gt')
    parser_compare.set_defaults(func=compare)

    parser_m2g = subparsers.add_parser('m2g')
    parser_m2g.add_argument("--mask", type=str, help="mask path")
    parser_m2g.add_argument("--object_type", type=str, default='annotation')
    parser_m2g.add_argument("--connectivity", type=int, default=4)
    parser_m2g.add_argument("--transform", type=str, default=None)
    parser_m2g.add_argument("--downsample", type=float, default=1.0)
    parser_m2g.add_argument("--include_labels", type=bool, default=False)
    parser_m2g.add_argument("--classification", type=str, default=None)
    parser_m2g.add_argument('-o', "--out", type=str, default=None, help="path for geojson file")
    parser_m2g.set_defaults(func=m2g)
    return parser.parse_args(args)

# ==================== #
#         MAIN         #
# ==================== #
def main(args=None):
    args = parse_args(args)
    args.func(args)



if __name__ == '__main__':
    sys.exit(main())