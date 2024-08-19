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
        props = dict(object_type=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'CellID', 'value': s[1]}]
            
        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = classification
        
        # Wrap in a dict to effectively create a GeoJSON Feature
        po = dict(type="Feature", geometry=s[0], properties=props)

        features.append(po)
    
    return features

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
        json.dump(features, f)

# with open("../orion/fichier_test/240523_POCIJ_multi_2024-05-23-10-18_P102_CYTPIC_A31_C99e_full_000201.geojson") as filein, open("../cytpic.geojson", "w") as fileout:
#     chunk = json_buffer(filein)
#     while not chunk:
#         _ = fileout.write(p.sub("detection", chunk))
#         chunk = json_buffer(filein)

# def json_buffer(stream):
#     c = stream.read(1)
#     res = c
#     while c != "}":
#         c = stream.read(1)
#         res += c
#         if not c:
#             break
#     return res