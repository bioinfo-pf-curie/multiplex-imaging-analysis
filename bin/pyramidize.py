#!/usr/bin/env python

import argparse
import pathlib

import tifffile
from palom.pyramid import (normalize_mosaics, count_num_channels, PyramidSetting, 
                           tile_from_combined_mosaics, tile_from_pyramid)
import palom.reader
from utils import OmeTifffile


def detect_pixel_size(metadata):
    try:
        pixel_size = metadata.pix.physical_size_x
    except Exception as err:
        print(err)
        print('\nPixel size detection using ome-types failed')
        pixel_size = None
    return pixel_size


def _file(path):
    path = pathlib.Path(path)
    if path.is_file(): return path
    else: raise FileNotFoundError(path.resolve())

# rewrite of this function from palom because metadata can't be passed all completly with tifffile without indicated shaped=False
# as its necessary to write custom ome tiff with tifffile (with option ome=True tifffile will write its info and not mine...)
def write_pyramid(
    mosaics,
    output_path,
    downscale_factor=4,
    compression=None,
    is_mask=False,
    tile_size=None,
    save_RAM=False,
    kwargs_tifffile=None
):
    """
    Write a multi resolution image from a tiff

    Parameters
    ----------

    mosaics: list of dask arrays
        input data (from palom.reader.OmePyramidReader(in_path).pyramid[0])

    output_path: Path or str
        output filename

    downscale_factor: int
        factor to diminish resolution of

    compression: str
        compression name 

    is_mask: bool
        flag if image is a mask file or not

    tile_size: tuple of int
        size of tile to work on

    save_RAM: bool
        if true, a other way will be use to save some RAM

    kwargs_tifffile: dict
        kwargs to be pass at tifffile.write
    """
    mosaics = normalize_mosaics(mosaics)
    ref_m = mosaics[0]
    num_channels = count_num_channels(mosaics)
    base_shape = ref_m.shape[1:3]
    assert int(downscale_factor) == downscale_factor
    assert downscale_factor < min(base_shape)
    pyramid_setting = PyramidSetting(
        downscale_factor=int(downscale_factor),
        tile_size=max(ref_m.chunksize)
    )
    num_levels = pyramid_setting.num_levels(base_shape)
    tile_shapes = pyramid_setting.tile_shapes(base_shape)
    shapes = pyramid_setting.pyramid_shapes(base_shape)

    if tile_size is not None:
        assert tile_size % 16 == 0, (
            f"tile_size must be None or multiples of 16, not {tile_size}"
        )
        tile_shapes = [(tile_size, tile_size)] * num_levels

    dtype = ref_m.dtype

    with tifffile.TiffWriter(output_path, bigtiff=True, shaped=False) as tif:
        if kwargs_tifffile is None:
            kwargs_tifffile = {}

        tif.write(
            data=tile_from_combined_mosaics(
                mosaics, tile_shape=tile_shapes[0], save_RAM=save_RAM
            ),
            shape=(num_channels, *shapes[0]),
            subifds=int(num_levels - 1),
            dtype=dtype,
            tile=tile_shapes[0],
            **kwargs_tifffile
        )
        
        for level, (shape, tile_shape) in enumerate(
            zip(shapes[1:], tile_shapes[1:])
        ):
            tif.write(
                data=tile_from_pyramid(
                    output_path,
                    num_channels,
                    tile_shape=tile_shape,
                    downscale_factor=downscale_factor,
                    level=level,
                    is_mask=is_mask,
                    save_RAM=save_RAM
                ),
                shape=(num_channels, *shape),
                subfiletype=1,
                dtype=dtype,
                tile=tile_shape,
                **{
                    **dict(compression=compression),
                    **kwargs_tifffile
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in',
        nargs='+',
        type=_file,
        required=True,
        help="Input Image Paths"
    )
    parser.add_argument('--out', type=str, required=False, help="Output Image Path")
    args = parser.parse_args()

    in_paths = vars(args)['in']
    # Automatically infer the output filename, if not specified
    if args.out is None:
        in_path = in_paths[0]
        stem = in_path.stem
        out_path = in_path.parent / f"{stem}.ome.tif"
    else:
        out_path = pathlib.Path(args.out)
    # pixel data is read into RAM lazily, cannot overwrite input file
    assert out_path not in in_paths

    metadata = OmeTifffile.from_path(in_paths[0])

    # Detect pixel size in ome-xml
    pixel_size = detect_pixel_size(metadata)
    if pixel_size is None: pixel_size = 1

    # Use palom to pyramidize the input image
    readers = [palom.reader.OmePyramidReader(in_path) for in_path in in_paths]
    mosaics = [reader.pyramid[0] for reader in readers]

    if max(mosaics[0].shape[1:3]) < 1024:
        # image is too small to compute sub resolution level
        with tifffile.TiffWriter(out_path, bigtiff=True, shaped=False) as tif:
            tif.write(
                data=mosaics[0],
                shape=mosaics[0].shape,
                **metadata.to_dict()
            )
    else:
        write_pyramid(mosaics, out_path, downscale_factor=2, kwargs_tifffile=metadata.to_dict(dtype=False))
