#!/usr/bin/env python

"""
Script for checking if image format is compatible with this pipeline (uncompressed ome-tiff)
uncompressed allow to better estimate RAM usage and quicker time 
perform conversion from all kind of other format
"""

import argparse
import tifffile
from ome_types import OME, model
import xml.etree.ElementTree as ET
from PIL import Image
import zarr

from utils import OmeTifffile, make_ome_data, read_tiff_orion, _tile_generator


def get_info_qptiff(tiff_mtd):
    for tag in tiff_mtd.tags:
        if tag.name == "ImageDescription":
            qptiff_data = ET.fromstring(tag.value)
            break
    else:
        raise ValueError('No qptiff data')
    
    version = qptiff_data.find('DescriptionVersion').text
    if version == "2":
        return qptiff2ome_v2(qptiff_data.find("ScanProfile")[0])
    elif version == "4":
        return qptiff2ome_v4(qptiff_data.find("ScanProfile"))

def qptiff2ome_v2(root):
    # PX = 0.325, PXU = µm, PY = 0.325, PYU = µm, PZ = 1, PZU=µm, size_c, size_t=1, size_z=1, size_x, size_y, dtype=uint16

    result = dict(
        PXU = "µm", PYU = "µm", PZU = "µm",
        PZ = 1, size_t=1, size_z=1,
        dtype="uint16"
    )

    for child in root:
        if "Resolution" in child.tag:
            result['PX'] = float(child.text)
            result['PY'] = result['PX']
            result["PXU"] = result['PYU'] = child.tag.rsplit('_', 1)[1]

    channels = []
    planes = []
    current_idx = 0
    for cycle in root.find('Cycles').findall('Cycle'):
        for channel in cycle.find('Channels').findall("Channel"):
            if channel.find('MarkerName').text.lower() not in ('empty', 'blank', ''):
                if "dapi" in channel.find('MarkerName').text.lower() and cycle.find('Index') != "1":
                    continue # do not add more than one dapi channel (other are used for alignment)
                channels.append(model.Channel(id=f"Channel:{current_idx}", name=channel.find('MarkerName').text, 
                                              samples_per_pixel=1, light_path=model.LightPath()))
                planes.append(model.Plane(the_c=current_idx, the_t=0, the_z=0))
                current_idx += 1

    result["size_c"] = len(channels)
    result['channels'] = channels
    result["planes"] = planes
    # when make_annotations is finished one should add "<AnnotationRef ID="Annotation:Stitcher:0"/>" before </Image>
    return result

def qptiff2ome_v4(root):
    result = dict(
        PXU = "µm", PYU = "µm", PZU = "µm",
        PZ = 1, size_t=1, size_z=1,
        dtype="uint16"
    )
    import json
    # !!!! Vulnerability !!!!
    wells = json.loads(root.text)['experimentDescription']['wells'] # new version (WIP)
    idx = 0
    channels = {}
    for well in wells:
        for item in well['items']:
            if item['markerName'] not in ('empty', 'blank', '', '--') and item['markerName'] not in channels: # do not add multiple channel with same name
                channels[item['markerName']] = idx
                idx += 1
    planes = [model.Plane(the_c=i, the_t=0, the_z=0) for i in channels.values()]
    channels = [model.Channel(id=f"Channel:{v}", name=k, samples_per_pixel=1, light_path=model.LightPath()) 
                for k, v in channels.items()]
    result["size_c"] = len(channels)
    result['channels'] = channels
    result["planes"] = planes
    # when make_annotations is finished one should add "<AnnotationRef ID="Annotation:Stitcher:0"/>" before </Image>
    return result

def hyperion2ome():
    pass

def guess_type():
    pass

def open_other_format(img_path):
    img = Image.open(img_path)
    # read metadata and populate default
    default_mtd = dict(
        size_x=img.height,
        size_y=img.width
    )
    if img.mode == 'I':
        default_mtd.update(dict(
            dtype='int32',
            size_c=img.n_frames
        ))
    elif img.mode == "RGB":
        default_mtd.update(dict(
            dtype="uint8",
            size_c=3
        ))
        # we need more mode
    else:
        default_mtd.update(dict(
            size_c=1,
            dtype="uint16"
        ))
    return img, default_mtd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, nargs='+', help="Input Image Path")
    parser.add_argument('--out', type=str, required=False, help="Output image Path")
    args = parser.parse_args()
    if len(args.image) > 1:
        # merging image 
        # hyperion ?
        img_path = "" # will be a new one or an in-memory
        pass
    else:
        img_path = args.image[0]

    # open Image
    if img_path.endswith("tiff") or img_path.endswith('tif'):
        try:
            img, mtd = read_tiff_orion(img_path)
            default_mtd = None
        except BaseException:
            img = tifffile.TiffFile(img_path)
            imgp = img.pages[0]
            img = zarr.open(img.series[0].aszarr())
            try:
                default_mtd = get_info_qptiff(imgp)
            except BaseException:
                dtype = [val for val in ('int8', 'int16', 'int32', 
                                         'uint8', 'uint16', 'uint32', 
                                         'float', 'double', 'complex', 
                                         'double-complex', 'bit') if val in str(img.dtype)]
                default_mtd = dict(
                    size_x=img.shape[-2],
                    size_y=img.shape[-1],
                    size_c=img.shape[0] if img.ndim == 3 else 1,
                    dtype=dtype[-1]
                )
            
            print(img)
    else:
        try:
            img, default_mtd = open_other_format(img_path)
        except BaseException:
            raise TypeError(f'Can not read image {img_path}. Unknown format')
        
    if default_mtd is not None:
        mtd = OmeTifffile()
        mtd.ome = make_ome_data(**default_mtd)
        mtd.dtype = default_mtd['dtype']

    mtd_dict = mtd.to_dict()

    # force no compression
    mtd_dict['compression'] = 1

    chunk_size = (4096,4096)
    img_shape = (mtd.pix.size_c, mtd.pix.size_y, mtd.pix.size_x)

    def tile_gen():
        for chan in range(mtd.pix.size_c):
            yield from _tile_generator(img, chan, mtd.pix.size_y, mtd.pix.size_x, *chunk_size)

    with tifffile.TiffWriter(args.out, bigtiff=True, shaped=False) as tif:
        tif.write(data=tile_gen(), shape=img_shape, tile=chunk_size, **mtd_dict)

