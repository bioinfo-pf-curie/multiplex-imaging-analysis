#!/usr/bin/env python

import argparse
import pandas as pd
import geojson
from shapely import Polygon, make_valid, geometry

CELLID = "CellID"
AREA = "Area"
SIZE_MIN = "minimal size"
SIZE_MAX = "maximal size"
NECROTIC = "Necrotic area"
AOI = "AOI"

def position_filter(df, geosjon_path):
    with open(geosjon_path, 'r') as gjfile:
        gj = geojson.load(gjfile)
    rois = []
    for roi in gj['features']:
        shapely_roi = make_valid(Polygon(roi.get('geometry', roi).get('coordinates')))
        if not isinstance(shapely_roi, Polygon):
            max_ = 0
            for g in shapely_roi.geoms:
                if max_ < g.area:
                    res = g
                    max_ = g.area
            shapely_roi = res
        rois.append(shapely_roi)
    
    rois = sorted(rois, key=lambda x: x.area, reverse=True) # the biggest one is the main one

    aoi = rois[0]
    for roi in rois[1:]:
        if aoi.intersects(roi): # exclusion
            aoi = aoi.difference(roi)
        else: # union
            aoi = aoi.union(roi)

    df['_point'] = df[['X_centroid', 'Y_centroid']].apply(geometry.Point, axis=1)
    df[AOI] = df['_point'].apply(aoi.contains)
    return df.drop('_point', axis=1)

def perform_filtering(csv, out_name, size_min=0, size_max=None, necrotic_intensity_treshold=0.9, roi_path=None):
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

    if roi_path is not None:
        df = position_filter(df, roi_path)

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
    parser.add_argument('--region_of_interest_path', type=str, required=False, 
                        help="path to a geojson describing a region of interest (can be the contour of a tumor for example)", default=1)
    args = parser.parse_args()

    perform_filtering(csv=args.csv_path, out_name=args.out_path, size_min=args.area_min, size_max=args.area_max, 
                      necrotic_intensity_treshold=args.necrotic_intensity_treshold, roi_path=args.region_of_interest_path)
