import tifffile
import zarr

def read_tiff_orion(img_path, idx_serie=0, idx_level=0, *args, **kwargs):
    tiff = tifffile.TiffFile(img_path, *args, **kwargs)
    zarray = zarr.open(tiff.series[idx_serie].aszarr())
    return (zarray[idx_level] if idx_level is not None and tiff.series[idx_level].is_pyramidal else zarray), tiff.pages[0]

def transfer_metadata(original_metadata, func='save'):
    result = {}
    other_props = ["ImageWidth", "ImageLength", "BitsPerSample", "SamplesPerPixel", 
                    "TileWidth", "TileLength", "TileOffsets", "TileByteCounts", "SubIFDs", "SampleFormat"]
    
    direct_props = ['PhotometricInterpretation',  "PlanarConfiguration", 'Compression', "Software", "ImageDescription"]

    def set_tag(tag, tag_name, param_name):
        if tag.name == tag_name:
            result[param_name] = tag.value

    for tag in original_metadata.tags:

        set_tag(tag, 'PhotometricInterpretation', "photometric")
        set_tag(tag, "PlanarConfiguration", "planarconfig")

        set_tag(tag, 'Compression', "compress" if func != 'write' else 'compression')
        
        set_tag(tag, "Software", "software")
        set_tag(tag, "ImageDescription", "description")

        if tag.name in ("XResolution", "YResolution", "ResolutionUnit"):
            if "resolution" not in result:
                result["resolution"] = [0,0,0]
            result["resolution"][("XResolution", "YResolution", "ResolutionUnit").index(tag.name)] = tag.value

        if tag not in direct_props:
            if 'extratags' not in result:
                result['extratags'] = []
            result['extratags'].append((tag.code, tag.dtype, tag.count, tag.value, True))

    return result
            

