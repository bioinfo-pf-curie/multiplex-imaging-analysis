#!/usr/bin/env python

import argparse
from utils import read_tiff_orion

MARKER = "marker_name"
SEGMENTATION = "segmentation"
NORMALIZATION = "normalization"
CYCLE = "cycle"

def generate(tiff_path, out_path, marker=None, not_seg=None, norm=None):
    _, mtd = read_tiff_orion(tiff_path)
    channels = [ch.name for ch in mtd.pix.channels]
    nc = len(channels)
    seg = [1] * nc
    for i in (not_seg or []):
        seg[i] = 0
    
    if marker is None:
        marker = [''] * nc
    if norm is None:
        norm = [''] * nc

    if len(channels) != len(marker):
        raise ValueError('Not same length for marker name and channel number in image')
    if len(channels) != len(seg):
        raise ValueError('Not same length for marker segmentation and channel number in image')
    if len(channels) != len(norm):
        raise ValueError('Not same length for marker normalization and channel number in image')

    result = f'{CYCLE},{MARKER},{SEGMENTATION},{NORMALIZATION}\n'

    for i, row in enumerate(zip(channels, marker, seg, norm)):
        result += f"{i},{row[1] or row[0]},{row[2]},{row[3]}\n"

    with open(out_path, 'w') as out:
        out.write(result)

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--out', type=str, required=False, default="panel.csv", help="Output filepath, default = 'panel.csv'")
    parser.add_argument('--names', type=str, required=False, help="ordered names of markers separated by a comma")
    parser.add_argument('--notSegmented', type=int, nargs="+", required=False, help="index of channels to be removed from segmentation")
    parser.add_argument('--normalization', type=str, required=False, help="ordered list of two number separated by a space for each marker separated by a comma" 
                        " that correspond to normalization (min-max) values")
    args = parser.parse_args()

    if not args.out.endswith(".csv"):
        args.out += ".csv"

    
    marker = [m for m in args.names.split(',')] if args.names is not None else None
    replacement = {'1': True, "0": False, "false": False, "true": True, "False": False, "non": False, "no": False} 
    # only need to replace false value cause every str will be evaluated to True
    # seg = [replacement.get(s, bool(s)) for s in args.segmentation.split(',')] if args.segmentation is not None else None
    try:
        norm = [f"{int(n.split(' ')[0])};{int(n.split(' ')[1])}" for n in args.normalization.split(',')] if args.normalization is not None else None
    except:
        norm = [f"{float(n.split(' ')[0])};{float(n.split(' ')[1])}" for n in args.normalization.split(',')] if args.normalization is not None else None
    
    generate(tiff_path=args.image, marker=marker, not_seg=args.notSegmented, norm=norm, out_path=args.out)
