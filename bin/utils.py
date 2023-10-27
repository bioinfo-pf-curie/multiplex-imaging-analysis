import tifffile
import zarr

def read_tiff_orion(img_path, idx_serie=0, idx_level=0, *args, **kwargs):
    tiff = tifffile.TiffFile(img_path, *args, **kwargs)
    zarray = zarr.open(tiff.series[idx_serie].aszarr())
    return (zarray[idx_level] if idx_level is not None and tiff.series[idx_level].is_pyramidal else zarray), tiff.pages[0]
