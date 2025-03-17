#!/usr/bin/env python

import argparse
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata import SpatialData
from anndata import AnnData
import pandas as pd

from utils import read_tiff_orion


COORDS_X = "X_centroid"
COORDS_Y = "Y_centroid"
INSTANCE_KEY = "CellID"


def ome2spatial_data(ome_path, out_path, mask_path=None, marker_info=None, quantif=None):
    return SpatialData(images=_get_images(), labels=_get_labels(mask_path), tables=_get_tables(quantif, marker_info))

def _get_images(ome_path):
    img, mtd = read_tiff_orion(ome_path)
    return Image2DModel.parse(img)

def _get_labels(mask_path):
    labels = read_tiff_orion(mask_path)
    return Labels2DModel.parse(labels)

def _get_tables(quantif, marker_info):
    tables_dict = {}
    markers = pd.read_csv(marker_info) if marker_info is not None else None
    markers.index = markers["marker_name"]
    coords = ["X_centroid", "Y_centroid"]

    for table_path in quantif:
        table_name = table_path.stem
        table = pd.read_csv(table_path)
        adata = AnnData(
            table[markers.index].to_numpy(),
            obs=table.drop(columns=markers.marker_name.tolist() + coords),
            var=markers,
            obsm={"spatial": table[coords].to_numpy()},
            dtype=float,
        )
        adata.obs["region"] = pd.Categorical([table_name] * len(adata))

        tables_dict[table_name] = TableModel.parse(
            adata, region=table_name, region_key="region", instance_key=INSTANCE_KEY
        )
    return tables_dict