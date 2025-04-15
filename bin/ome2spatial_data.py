#!/usr/bin/env python

import argparse
from pathlib import Path
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata import SpatialData
from anndata import AnnData
import pandas as pd
from dask_image.imread import imread


COORDS_X = "X_centroid"
COORDS_Y = "Y_centroid"
INSTANCE_KEY = "CellID"


def ome2spatial_data(ome_path, mask_path=None, marker_info=None, quantif=None):
    img_name = ome_path.stem
    if img_name.endswith('.ome'): img_name = img_name[:-4]

    kwargs = dict(images={img_name: _get_image(ome_path)})

    if mask_path is not None:
        kwargs['labels'] = {mask_path.stem: _get_label(mask_path)}

    if quantif is not None:
        kwargs['tables'] = {quantif.stem: _get_table(quantif, marker_info)}
    
    return SpatialData(**kwargs)

def _get_image(ome_path):
    return Image2DModel.parse(imread(ome_path))

def _get_label(mask_path):
    return Labels2DModel.parse(imread(mask_path).squeeze())

def _get_table(quantif, marker_info):
    if marker_info is not None:
        markers = pd.read_csv(marker_info)
        markers.index = markers["marker_name"]
    else:
        markers = None
    coords = ["X_centroid", "Y_centroid"]
    
    if quantif is None:
        return
    
    table = pd.read_csv(quantif)
    adata = AnnData(
        table[[col for col in table.columns if col not in [INSTANCE_KEY, COORDS_X, COORDS_Y]]].to_numpy(),
        obs=table[[INSTANCE_KEY]].astype(str),
        var=[col for col in table.columns if col not in [INSTANCE_KEY, COORDS_X, COORDS_Y]],
        obsm={"spatial": table[coords].to_numpy()},
        dtype=float,
    )
    adata.obs["region"] = pd.Categorical([quantif.stem] * len(adata))

    return TableModel.parse(
        adata, region=quantif.stem, region_key="region", instance_key=INSTANCE_KEY
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True, help="Ome Tiff file")
    parser.add_argument('--mask', type=Path, required=False, help="mask (output of segmentation) in tiff format")
    parser.add_argument('--out', type=str, help="Output directory")
    parser.add_argument('--quantification', type=Path, required=False, help="path to the quantification file")
    parser.add_argument('--panel', type=Path, required=False, help="path to the panel.csv")
    args = parser.parse_args()

    sp = ome2spatial_data(ome_path=args.image, mask_path=args.mask, marker_info=args.panel, quantif=args.quantification)
    sp.write(args.out)