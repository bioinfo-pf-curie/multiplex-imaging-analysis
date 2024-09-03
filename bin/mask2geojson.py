#!/usr/bin/env python

"""
Convert Python labeled image to QuPath-friendly GeoJSON.
"""

import rasterio.features
from rasterio.transform import Affine
import numpy as np
import json
import tifffile
import argparse

def mask2geojson(mask: np.ndarray, object_type='detection', connectivity: int=8, 
                 transform: Affine=None, downsample: float=1.0, include_labels=False,
                 classification=None):
    """
    Create a GeoJSON FeatureCollection from a labeled image.
    """
    features = []
    
    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(downsample)
    
    # Trace geometries
    for s in rasterio.features.shapes(mask, mask=mask > 0, 
                                      connectivity=connectivity, transform=transform):

        # Create properties
        props = dict(objectType=object_type)
        if include_labels:
            props['measurements'] = {'CellID': int(s[1])}
            
        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = classification
        
        # Wrap in a dict to effectively create a GeoJSON Feature
        po = dict(type="Feature", geometry=s[0], properties=props)

        features.append(po)
    
    return {"type": "FeatureCollection", "features": features}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', type=str, required=True, help="filename of mask")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting geojson file")
    args = parser.parse_args()

    # read image
    mask = tifffile.imread(args.mask)
    mask = mask.astype('float32')

    # Create GeoJSON-like version
    features = mask2geojson(mask, include_labels=True)

    # Convert to GeoJSON string
    with open(args.out, "w") as f:
        json.dump(features, f, separators=(",", ":"))


"""
def read_part(start=0, length=100000):
    with open("orion/fichier_test/240523_POCIJ_mesmer.geojson") as geojson:
        geojson.seek(start)
        return geojson.read(length)

def convert_json(dfg, start_char, stop_char):
    return json.loads("[" + dfg[start_char: stop_char] + "]")

def handle_unique(data, seen=None):
    seen = seen or []
    for poly in data:
        pid = poly["properties"]['measurements'][0]["value"]
        if pid not in seen:
            seen.append(pid)
        else:
            print(f"{pid} has been found")
            break
    return seen

"""