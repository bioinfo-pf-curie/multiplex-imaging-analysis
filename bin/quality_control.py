#!/usr/bin/env python

import argparse
import pandas as pd

CELLID = "CellID"
AREA = "Area"
SIZE_MIN = "minimal size"
SIZE_MAX = "maximal size"
NECROTIC = "Necrotic area"

def perform_filtering(csv, out_name, size_min=0, size_max=None, necrotic_intensity_treshold=0.9):
    df = pd.read_csv(csv)

    form_cols = (
        "X_centroid",
        "Y_centroid",
        "column_centroid",
        "row_centroid",
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "Solidity",
        "Extent",
        "Orientation",
    )

    markers_cols = [c for c in df.columns if (c not in form_cols) and (c != CELLID)]

    # size filtering
    if size_max is None:
        size_max = df[AREA].max()
    df[f'{SIZE_MIN} ({size_min})'] = (size_min < df[AREA]).astype(int)
    df[f'{SIZE_MAX} ({size_max})'] = (df[AREA] <= size_max).astype(int)

    # necrotic filtering
    df[f'{NECROTIC} ({necrotic_intensity_treshold}% intensity)'] = ~(
        df[markers_cols] > df[markers_cols].quantile(necrotic_intensity_treshold)
    ).all(axis=1).astype(int)

    df.to_csv(out_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, required=True, help="original image path")
    # parser.add_argument('--mask', type=str, required=True, help="mask path")
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--out_path', type=str, required=True, help="output path")
    parser.add_argument('--area_min', type=int, default=0, required=False, help="minimal cell area")
    parser.add_argument('--area_max', type=int, required=False, help="maximal cell area")
    parser.add_argument('--necrotic_intensity_treshold', type=float, required=False, 
                        help="treshold of intensity (normalized between 0 and 1) "
                             "for a cell to be considered as necrotic (in every markers)", default=1)
    args = parser.parse_args()

    perform_filtering(csv=args.csv_path, out_name=args.out_path, size_min=args.area_min, size_max=args.area_max, 
                      necrotic_intensity_treshold=args.necrotic_intensity_treshold)