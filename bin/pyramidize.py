#!/usr/bin/env python
# some function are from https://github.com/labsyspharm/palom
import logging
import argparse
import pathlib
import math
import gc
import itertools

import tifffile
import numpy as np
import dask.array as da
import cv2
import zarr
import tqdm
import ome_types
import pint

from utils import OmeTifffile


logger = logging.getLogger(__name__)


class DaPyramidChannelReader:
    def __init__(self, pyramid: list[da.Array], channel_axis: int) -> None:
        self.pyramid = pyramid
        self.channel_axis = channel_axis
        if self.validate_pyramid(self.pyramid, self.channel_axis):
            self.pyramid = self.normalize_axis_order()
            self.pyramid = self.auto_format_pyramid(self.pyramid)

    @staticmethod
    def validate_pyramid(pyramid: list[da.Array], channel_axis: int) -> bool:
        for i, level in enumerate(pyramid):
            assert level.ndim == 3
            if np.argmin(level.shape) != channel_axis:
                logger.warning(
                    f"level {i} has shape of {level.shape} while given"
                    f" `channel_axis` is {channel_axis}"
                )
        return True

    def normalize_axis_order(self):
        if self.channel_axis == 0:
            return self.pyramid
        return [da.moveaxis(level, self.channel_axis, 0) for level in self.pyramid]

    def read_level_channels(self, level: int, channels: int | list[int]) -> da.Array:
        target_level = self.pyramid[level]
        return target_level[channels]

    @staticmethod
    def auto_format_pyramid(
        pyramid: list[da.Array],
    ) -> list[da.Array]:
        first = pyramid[0]
        if len(pyramid) > 1:
            return pyramid
        # Assumption: if the image is pyramidal, it must also be tiled
        if max(first.shape) < 1024:
            return pyramid
        logger.warning(
            "Unable to detect pyramid levels, it may take a while"
            " to compute thumbnails during coarse alignment"
        )
        if first.numblocks[1:3] == (1, 1):
            first = first.rechunk((1, 1024, 1024))
        pyramid_setting = PyramidSetting(downscale_factor=2)
        num_levels = pyramid_setting.num_levels(first.shape[1:3])
        return [
            da.coarsen(
                np.mean, first, {0: 1, 1: 2**i, 2: 2**i}, trim_excess=True
            ).astype(first.dtype)
            for i in range(num_levels)
        ]

    @property
    def level_downsamples(self) -> dict[int, float]:
        shapes = [ss.shape[1:3] for ss in self.pyramid]
        shapes.insert(0, shapes[0])
        # FIXME should use image-based registration to further refine the
        # downsample factor between levels
        downsamples = [
            np.divide(s1, s2).mean() for s1, s2 in itertools.pairwise(shapes)
        ]
        return dict(enumerate(itertools.accumulate(downsamples, func=np.multiply)))

    @property
    def pixel_dtype(self) -> np.dtype:
        return self.pyramid[0].dtype

    def get_thumbnail_level_of_size(self, size: float) -> int:
        shapes = [np.abs(np.mean(level.shape[1:3]) - size) for level in self.pyramid]
        return np.argmin(shapes)


class OmePyramidReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, pixel_size: float | None = None
    ) -> None:
        self.path = pathlib.Path(path)
        pyramid = self.pyramid_from_ometiff(self.path)
        channel_axis = 0
        self._pixel_size = pixel_size
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state["path"], pixel_size=state["_pixel_size"])

    @staticmethod
    def pyramid_from_ometiff(path: str | pathlib.Path) -> list[da.Array]:
        with tifffile.TiffFile(path) as tif:
            num_series = len(tif.series)
            if num_series == 1:
                pyramid = tif.series[0].levels
            elif num_series > 1:
                pyramid = tif.series
            zarr_pyramid = [zarr.open(level.aszarr(), mode="r") for level in pyramid]
            da_pyramid = []
            for z in zarr_pyramid:
                if issubclass(type(z), zarr.Group):
                    da_level = da.from_zarr(z[0], name=False)
                else:
                    da_level = da.from_zarr(z, name=False)
                da_level = da_level.squeeze()
                if da_level.ndim == 2:
                    da_level = da_level.reshape(1, *da_level.shape)
                elif da_level.ndim == 3:
                    if da_level.shape[2] in (3, 4):
                        da_level = da.moveaxis(da_level, 2, 0)
                else:
                    raise ValueError(
                        f"Image with {da_level.ndim} dimension {da_level.shape} is not supported"
                    )
                da_pyramid.append(da_level)
        return da_pyramid

    @property
    def pixel_size(self) -> float:
        if self._pixel_size is not None:
            return self._pixel_size
        try:
            # ome-types v0.4 does not have `parser` kwarg in `from_tiff`
            import inspect

            kwargs = dict(path=self.path, validate=False)
            keys = inspect.signature(ome_types.from_tiff).parameters
            if "parser" in keys:
                kwargs.update(dict(parser="lxml"))
            ome = ome_types.from_tiff(**kwargs)
            px_size = ome.images[0].pixels.physical_size_x
            # convert length unit to µm
            unit = ome.images[0].pixels.physical_size_x_unit.value
            ureg = pint.UnitRegistry()
            px_size_micron = px_size * ureg(unit).to(ureg.micron).magnitude
            logger.info(f"Detected pixel size: {px_size_micron:.4f} µm")
            self._pixel_size = px_size_micron
            return self._pixel_size
        except Exception:
            logger.warning(
                f"Unable to parse pixel size from {self.path.name};"
                f" assuming 1 µm. Use `_pixel_size` to set it manually"
            )
            self._pixel_size = 1
            return self._pixel_size


def count_num_channels(imgs):
    for img in imgs:
        assert img.ndim == 2 or img.ndim == 3
    return sum([1 if img.ndim == 2 else img.shape[0] for img in imgs])


class PyramidSetting:
    def __init__(self, downscale_factor=2, tile_size=1024, max_pyramid_img_size=1024):
        self.downscale_factor = downscale_factor
        self.tile_size = tile_size
        self.max_pyramid_img_size = max_pyramid_img_size

    def tile_shapes(self, base_shape):
        shapes = np.array(self.pyramid_shapes(base_shape))
        n_rows_n_cols = np.ceil(shapes / self.tile_size)
        tile_shapes = np.ceil(shapes / n_rows_n_cols / 16) * 16
        return [tuple(map(int, s)) for s in tile_shapes]

    def pyramid_shapes(self, base_shape):
        num_levels = self.num_levels(base_shape)
        factors = self.downscale_factor ** np.arange(num_levels)
        shapes = np.ceil(np.array(base_shape) / factors[:, None])
        return [tuple(map(int, s)) for s in shapes]

    def num_levels(self, base_shape):
        factor = max(base_shape) / self.max_pyramid_img_size
        return max(math.ceil(math.log(factor, self.downscale_factor)) + 1, 1)


def normalize_mosaics(mosaics, tile_size=None):
    dtypes = set(m.dtype for m in mosaics)
    if any([np.issubdtype(d, np.floating) for d in dtypes]):
        max_dtype = np.dtype(np.float32)
    else:
        max_dtype = max(dtypes)
    if tile_size is None:
        tile_size = 1024
    normalized = []
    for m in mosaics:
        assert m.ndim == 2 or m.ndim == 3
        if m.ndim == 2:
            m = m[np.newaxis, :]
        if not isinstance(m, da.core.Array):
            m = da.from_array(m, chunks=(1, tile_size, tile_size), name=False)
        normalized.append(m.astype(max_dtype, copy=False))
    return normalized


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


def tile_from_combined_mosaics(mosaics, tile_shape, save_RAM=False):
    num_rows, num_cols = mosaics[0].shape[1:3]
    h, w = tile_shape
    n = len(mosaics)
    for idx, m in enumerate(mosaics):
        for cidx, c in enumerate(m):
            # the performance is heavily degraded without pre-computing the
            # mosaic channel
            with tqdm.dask.TqdmCallback(
                ascii=True,
                desc=(
                    f"Assembling mosaic {idx + 1:2}/{n:2} (channel"
                    f" {cidx + 1:2}/{m.shape[0]:2})"
                ),
            ):
                c = da_to_zarr(c) if save_RAM else c.compute()
            for y in range(0, num_rows, h):
                for x in range(0, num_cols, w):
                    yield np.array(c[y : y + h, x : x + w])
                    # yield m[y:y+h, x:x+w].copy().compute()
            c = None


def tile_from_pyramid(
    path,
    num_channels,
    tile_shape,
    downscale_factor=2,
    level=0,
    is_mask=False,
    save_RAM=False,
):
    # workaround progress bar
    # https://forum.image.sc/t/tifffile-ome-tiff-generation-is-taking-too-much-ram/41865/26
    pbar = tqdm.tqdm(total=num_channels, ascii=True, desc="Processing channel")
    for c in range(num_channels):
        gc.collect()
        img = da.from_zarr(
            zarr.open(
                tifffile.imread(path, series=0, level=level, aszarr=True), mode="r"
            ),
            name=False,
        )
        if img.ndim == 2:
            img = img.reshape(1, *img.shape)
        img = img[c]
        # read using key seems to generate a RAM spike
        # img = tifffile.imread(path, series=0, level=level, key=c)
        if not is_mask:
            img = img.map_blocks(
                cv2.blur, ksize=(downscale_factor, downscale_factor), anchor=(0, 0)
            )
        img = da_to_zarr(img) if save_RAM else img.compute()
        num_rows, num_columns = img.shape
        h, w = tile_shape
        h *= downscale_factor
        w *= downscale_factor
        last_c = range(num_channels)[-1]
        last_y = range(0, num_rows, h)[-1]
        last_x = range(0, num_columns, w)[-1]
        for y in range(0, num_rows, h):
            for x in range(0, num_columns, w):
                if (y == last_y) & (x == last_x):
                    pbar.update(1)
                    if c == last_c:
                        pbar.close()
                yield np.array(
                    img[y : y + h : downscale_factor, x : x + w : downscale_factor]
                )
        # setting img to None seems necessary to prevent RAM spike
        img = None


def da_to_zarr(da_img, zarr_store=None, num_workers=None, out_shape=None, chunks=None):
    if zarr_store is None:
        if out_shape is None:
            out_shape = da_img.shape
        if chunks is None:
            chunks = da_img.chunksize
        zarr_store = zarr.create(
            out_shape, chunks=chunks, dtype=da_img.dtype, overwrite=True
        )
    da_img.to_zarr(zarr_store, compute=False).compute(num_workers=num_workers)
    return zarr_store


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
                **kwargs_tifffile
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
    parser.add_argument('--compression', type=int, required=False, help="tifffile compression name possible value are 'none', 'zlib', 'jpeg', ... see doc. Can raise error if its not compatible with other parms")
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

    # Use palom to pyramidize the input image
    readers = [OmePyramidReader(in_path) for in_path in in_paths]
    mosaics = [reader.pyramid[0] for reader in readers]

    # Detect pixel size in ome-xml
    try:
        metadata = OmeTifffile.from_path(in_paths[0])
    except:
        img_shape = mosaics[0].shape
        metadata = OmeTifffile(size_c=img_shape[0], size_x=img_shape[1], size_y=img_shape[2])

    pixel_size = detect_pixel_size(metadata)
    if pixel_size is None: pixel_size = 1

    if max(mosaics[0].shape[1:3]) < 1024:
        # image is too small to compute sub resolution level
        with tifffile.TiffWriter(out_path, bigtiff=True, shaped=False) as tif:
            tif.write(
                data=mosaics[0],
                shape=mosaics[0].shape,
                **metadata.to_dict()
            )
    else:
        kwargs = metadata.to_dict(dtype=False)
        if args.compression:
            kwargs.update(compression=args.compression)
        write_pyramid(mosaics, out_path, downscale_factor=2, save_RAM=True, kwargs_tifffile=kwargs)
