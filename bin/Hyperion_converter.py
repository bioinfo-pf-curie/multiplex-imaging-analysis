#!/usr/bin/env python
from readimc import MCDFile

# MCD file are organized in Slides Panoramas and Acquisitions, 
# Not sure what are they exactly but slide and panoramas seems to be in RGB and acquisition with the correct channels numbers but not complete ??

def read_acquisition(f, acquisition):
    """rewrite this function to handle memmap and buffer (not loading all that in memory...)"""
    if acquisition is None:
        raise ValueError("acquisition")
    if f._fh is None:
        raise IOError(f"MCD file '{f.path.name}' has not been opened")
    try:
        data_start_offset = int(acquisition.metadata["DataStartOffset"])
        data_end_offset = int(acquisition.metadata["DataEndOffset"])
        value_bytes = int(acquisition.metadata["ValueBytes"])
    except (KeyError, ValueError) as e:
        raise IOError(
            f"MCD file '{f.path.name}' corrupted: "
            "cannot locate acquisition image data"
        ) from e
    if data_start_offset >= data_end_offset:
        raise IOError(
            f"MCD file '{f.path.name}' corrupted: "
            "invalid acquisition image data offsets"
        )
    if value_bytes <= 0:
        raise IOError("MCD file corrupted: invalid byte size")
    num_channels = acquisition.num_channels
    data_size = data_end_offset - data_start_offset
    bytes_per_pixel = (num_channels + 3) * value_bytes
    if data_size % bytes_per_pixel != 0:
        data_size += 1
    if data_size % bytes_per_pixel != 0:
        if strict:
            raise IOError(
                f"MCD file '{f.path.name}' corrupted: "
                "invalid acquisition image data size"
            )
        warn(
            f"MCD file '{f.path.name}' corrupted: "
            "invalid acquisition image data size"
        )
    num_pixels = data_size // bytes_per_pixel
    f._fh.seek(0)
    return np.memmap(
        f._fh,
        dtype=np.float32,
        mode="r",
        offset=data_start_offset,
        shape=(num_pixels, num_channels + 3),
    )

def mcd2ometiff(image_path):
    with MCDFile(image_path) as f:
        pass