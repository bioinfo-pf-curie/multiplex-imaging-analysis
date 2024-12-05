#!/usr/bin/env python

import argparse
import pandas as pd
import json
from shapely import Polygon, make_valid, geometry

CELLID = "CellID"
AREA = "Area"
SIZE_MIN = "minimal size"
SIZE_MAX = "maximal size"
NECROTIC = "Necrotic area"
AOI_IN = "RoI"
AOI_OUT = "Exclusion"

def position_filter(points, geosjon_path):
    with open(geosjon_path, 'r') as gjfile:
        gj = json.load(gjfile)

    res = pd.Series(index=points.index, dtype="str")

    for i, roi in enumerate(gj['features'], 1):
        shapely_roi = make_valid(Polygon(roi.get('geometry', roi).get('coordinates')))
        if not isinstance(shapely_roi, Polygon):
            max_ = 0
            for g in shapely_roi.geoms:
                if max_ < g.area:
                    res = g
                    max_ = g.area
            shapely_roi = res
        roi_name = roi.get('properties', roi).get('classification', {}).get('name', str(i))

        inter = points.apply(shapely_roi.contains) 

        res.loc[inter & ~res.isna()] += f", {roi_name}"
        res.loc[inter & res.isna()] = roi_name
    return res

def perform_filtering(csv, out_name, size_min=None, size_max=None, 
                      necrotic_intensity_treshold=None, roi_path=None, excluded_path=None):
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
    if size_max is not None:
        df[f'{SIZE_MAX} ({int(size_max)})'] = (df[AREA] <= size_max).astype(int)
    if size_min is not None:
        df[f'{SIZE_MIN} ({int(size_min)})'] = (size_min < df[AREA]).astype(int)

    # necrotic filtering
    if necrotic_intensity_treshold is not None:
        df[f'{NECROTIC} ({necrotic_intensity_treshold}% intensity)'] = ~(
            df[markers_cols] > df[markers_cols].quantile(necrotic_intensity_treshold)
        ).all(axis=1).astype(int)

    if roi_path is not None or excluded_path is not None:
        points = df[['X_centroid', 'Y_centroid']].apply(geometry.Point, axis=1)
        if roi_path is not None:
            df[AOI_IN] = position_filter(points, roi_path)
        if excluded_path is not None:
            df[AOI_OUT] = position_filter(points, excluded_path)


    df.to_csv(out_name, index=False)


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
    parser.add_argument('--region_of_interest_geojson_path', type=str, required=False, 
                        help="path to a geojson describing a region of interest  or a list of (can be the contour of a tumor for example)")
    parser.add_argument('--excluded_region_geojson_path', type=str, required=False, 
                        help="path to a geojson describing a (or a list of) region to exclude (dificult to segment for example)")
    args = parser.parse_args()

    perform_filtering(csv=args.csv_path, out_name=args.out_path, size_min=args.area_min, size_max=args.area_max, 
                      necrotic_intensity_treshold=args.necrotic_intensity_treshold,
                      roi_path=args.region_of_interest_geojson_path, excluded_path=args.excluded_region_geojson_path)
