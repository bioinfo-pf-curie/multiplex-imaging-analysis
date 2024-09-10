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

def mask2geojson(mask: np.ndarray, object_type='detection', connectivity: int=4, 
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
import json_stream
geojson_path = "orion/fichier_test/240523_POCIJ_15diam.geojson"
with open(geojson_path) as f:
    data = json_stream.load(f)
    total = []
    for polygon in data['features']:
        # geom = polygon['geometry']['coordinates'] # its written first in dict, so if we want that we need to acces it first in stream
        total.append(polygon['properties']['measurements']['CellID'])
print(len(total))
print(len(set(total)))


with open(geojson_path) as f:
    data = json_stream.load(f)
    total = []
    for polygon in data['features']:
            geom = json_stream.to_standard_types(polygon['geometry']['coordinates'])
            if polygon['properties']['measurements']['CellID'] == double[0]:
                    total.append(geom)
            if len(total) > 1:
                    break

"""