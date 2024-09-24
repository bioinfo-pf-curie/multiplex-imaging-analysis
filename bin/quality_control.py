#!/usr/bin/env python

import argparse
import pandas as pd

CELLID = "CellID"
AREA = "Area"

def perform_filtering(csv, out_name, size_min=None, size_max=None, necrotic_intensity_treshold=0.9):
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
    df = df.loc[(size_min or 0) < df[AREA] <= (size_max or df[AREA].max())]

    # necrotic filtering
    df = df.loc[~(df[markers_cols] > df[markers_cols].quantile(necrotic_intensity_treshold)).all(axis=1)]

    df.to_csv(out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, required=True, help="original image path")
    # parser.add_argument('--mask', type=str, required=True, help="mask path")
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--out_path', type=str, required=True, help="output path")
    parser.add_argument('--area_min', type=int, required=False, help="minimal cell area")
    parser.add_argument('--area_max', type=int, required=False, help="maximal cell area")
    parser.add_argument('--necrotic_intensity_treshold', type=float, required=False, 
                        help="treshold of intensity (normalized between 0 and 1) "
                             "for a cell to be considered as necrotic (in every markers)")
    args = parser.parse_args()

    perform_filtering(csv=args.csv_path, out_name=args.out_path, size_min=args.area_min, size_max=args.area_max, 
                      necrotic_intensity_treshold=args.necrotic_intensity_treshold)