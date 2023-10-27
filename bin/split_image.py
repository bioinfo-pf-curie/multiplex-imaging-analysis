#!/usr/bin/env python

import os
import sys
from tifffile import TiffWriter

from utils import read_tiff_orion


def split_img(img_path, height=224, overlap=50):
    """Will split an image into height x image_width crop (with some overlap) to get a better memory footprint """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    img_zarr, metadata = read_tiff_orion(img_path)
    ch, total_height, total_width = img_zarr.shape
    for cur_height in range(0, total_height, height - overlap):
        out_path = img_name + f"_{cur_height}" + ext
        with TiffWriter(out_path, ome=True, bigtiff=True) as tiff_out:
            tmp_arr = img_zarr[:, cur_height: cur_height+height, :]
            tiff_out.write(
                data=tmp_arr,
                software=metadata.software,
                shape=tmp_arr.shape,
                #subifds=int(self.num_levels - 1),
                dtype=metadata.dtype,
                resolution=(
                    metadata.tags["XResolution"].value,
                    metadata.tags["YResolution"].value,
                    metadata.tags["ResolutionUnit"].value),
                photometric=metadata.photometric,
                compression="adobe_deflate",
                predictor=True,
            )


if __name__ == "__main__":
    split_img(sys.argv[1])