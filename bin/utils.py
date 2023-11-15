from typing import ClassVar, Any

import tifffile
import zarr
from ome_types import OME, model

def read_tiff_orion(img_path, idx_serie=0, idx_level=0, *args, **kwargs):
    tiff = tifffile.TiffFile(img_path, *args, **kwargs)
    zarray = zarr.open(tiff.series[idx_serie].aszarr())
    return (zarray[idx_level] if idx_level is not None and tiff.series[idx_level].is_pyramidal else zarray), OmeTifffile(tiff.pages[0])

class OmeTifffile(object):
    direct_props = {'PhotometricInterpretation': "photometric",  
                    "PlanarConfiguration": "planarconfig", 
                    'Compression': "compress", 
                    "Software": "software"}

    def __init__(self, tifffile_metadata, **kwargs):
        self.tags = {"resolution": [None, None, None], "extratags": []}

        for t in tifffile_metadata.tags:
            if t.name == "ImageDescription":
                self.ome = OME.from_xml(t.value, **kwargs)
                break
        else:
            raise ValueError("No image description tag was found")
        
        for tag in tifffile_metadata.tags:
            if t.name == "ImageDescription":
                continue

            if tag.name in self.direct_props.keys():
                self.tags[self.direct_props[tag.name]] = tag.value

            elif tag.name in ("XResolution", "YResolution", "ResolutionUnit"):
                self.tags["resolution"][("XResolution", "YResolution", "ResolutionUnit").index(tag.name)] = tag.value

            else:
                self.tags['extratags'].append((tag.code, tag.dtype, tag.count, tag.value, True)) # true for tag.writeonce (orion is one image per tiff)
    
        self.dtype = tifffile_metadata.dtype

    @property
    def fimg(self):
        return self.ome.images[0]
    
    @fimg.setter
    def fimg(self, value):
        self.ome.images[0] = value
    
    @property
    def pix(self):
        return self.fimg.pixels
    
    @pix.setter
    def pix(self, value):
        self.fimg.pixels = value

    def to_dict(self, func_name="write"):
        this_dict = self.tags.copy()
        if func_name == 'save':
            this_dict['compression'] = this_dict.pop('compress')
        this_dict['description'] = self.ome.to_xml()
        if any(resolution is None for resolution in self.tags["resolution"]):
            # was not set
            this_dict.pop('resolution')
        return this_dict
    
    def add_channel(self, channel_data):
        self.pix.channels.append(channel_data)
        self.pix.planes.append({'the_z': 0, 'the_t': 0, 'the_c': int(self.pix.size_c)})
        self.pix.size_c = len(self.pix.channels)
    
    def add_channel_metadata(self, channel_name, add_prefix=True, **kwargs):
        if 'samples_per_pixel' not in kwargs:
            kwargs['samples_per_pixel'] = 1
        if 'light_path' not in kwargs:
            kwargs['light_path'] = {}

        try:
            old_id = self.pix.channels[-1].id
            new_id = int(old_id.split(':')[-1]) + 1
        except (IndexError, ValueError, AttributeError):
            new_id = len(self.pix.channels)
        new_id_name = f"Channel:{new_id}"

        if add_prefix:
            try:
                prefix = int(self.pix.channels[-1].name.split('_')[0]) + 1
            except (IndexError, ValueError, AttributeError):
                prefix = 1
            channel_name = f"{prefix:02d}_{channel_name}"
        
        self.add_channel(model.Channel(
            id=new_id_name, name=channel_name, **kwargs
        ))

    def remove_all_channels(self):
        self.pix.planes = []
        self.pix.channels = []
        self.pix.size_c = 1 # can't put 0 validation error

    def get_channel(self, id_chan):
        return self.pix.channels[id_chan]
        
"""
img_path = "/data/users/mcorbe/orion/data/2017206/220516_Lung_18p_P39_A28_C76dX_E16_Curie18@20220519_110224_090101.ome.tiff"
import tifffile
from orion.MIA.bin.utils import OmeTifffile
info = tifffile.TiffFile(img_path)
o = OmeTifffile(info.pages[0])

"""