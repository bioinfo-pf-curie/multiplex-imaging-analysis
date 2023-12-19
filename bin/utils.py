import tifffile
import zarr
from ome_types import OME, model
import copy


def _tile_generator(arr, channel, x, y, chunk_x, chunk_y):
    for x_cur in range(0, x, chunk_x):
        for y_cur in range(0, y, chunk_y):
            yield arr[channel, x_cur: x_cur + chunk_x, y_cur: y_cur + chunk_y]


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
        self.ome = None

        for tag in tifffile_metadata.tags:
            if tag.name == "ImageDescription":
                self.ome = OME.from_xml(tag.value, **kwargs)

            elif tag.name in self.direct_props.keys():
                try:
                    self.tags[self.direct_props[tag.name]] = tag.value.value
                except AttributeError:
                    self.tags[self.direct_props[tag.name]] = tag.value

            elif tag.name in ("XResolution", "YResolution", "ResolutionUnit"):
                self.tags["resolution"][("XResolution", "YResolution", "ResolutionUnit").index(tag.name)] = tag.value

            else:
                self.tags['extratags'].append((tag.code, tag.dtype, tag.count, tag.value, True)) # true for tag.writeonce (orion is one image per tiff)

        if self.ome is None:
            raise ValueError("No image description tag was found")
            
        self.dtype = tifffile_metadata.dtype

    @classmethod
    def from_path(cls, tiff_path):
        _, mtd = read_tiff_orion(tiff_path)
        return mtd

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

    def update_shape(self, arr_shape, order="CYX"):
        for idx, char in enumerate(order):
            self.pix.__setattr__(f"size_{char.lower()}", arr_shape[idx])

    def to_dict(self, dtype=True):
        this_dict = self.tags.copy()
        this_dict['compression'] = this_dict.pop('compress')

        if any(resolution is None for resolution in self.tags["resolution"]):
            # was not set
            this_dict.pop('resolution')

        for key in this_dict:
            if type(this_dict[key]) is dict:
                this_dict[key] = str(this_dict[key])
        
        this_dict['description'] = self.ome.to_xml().encode()
        if dtype:
            this_dict['dtype'] = self.dtype

        return this_dict

    def add_channel(self, channel_data):
        # sometimes planes are not registered here
        if len(self.pix.planes) == len(self.pix.channels):
            self.pix.planes.append(model.Plane(the_z=0, the_t=0, the_c=int(self.pix.size_c)))
        try:
            self.pix.tiff_data_blocks[0].plane_count += 1
        except IndexError:
            self.pix.tiff_data_blocks.append(model.TiffData(plane_count=len(self.pix.planes)))
        except BaseException as e: 
            print("add channel error")
            print(e)
        self.pix.channels.append(channel_data)
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
        try:
            self.pix.tiff_data_blocks[0].plane_count = 0
        except BaseException as e: 
            print(e)

    def get_channel(self, id_chan):
        return self.pix.channels[id_chan]
    
    def copy(self):
        return copy.deepcopy(self)
        
"""
img_path = "/data/users/mcorbe/orion/data/2017206/220516_Lung_18p_P39_A28_C76dX_E16_Curie18@20220519_110224_090101.ome.tiff"
import tifffile
from orion.MIA.bin.utils import OmeTifffile
info = tifffile.TiffFile(img_path)
o = OmeTifffile(info.pages[0])

from orion.MIA.bin.utils import read_tiff_orion
from tifffile import TiffWriter
img_path = "orion/data/2017206/220513_Kidney_18p_P39_A28_C76dX_E16_Curie18@20220513_112541_663280.ome.tiff"
img, metadata = read_tiff_orion(img_path)

with TiffWriter("autre_tile.ome.tiff", bigtiff=True, shaped=False) as tiff_out:
    tmp_arr = img[:, 5000:6024, 5000:6024]
    metadata.update_shape(tmp_arr.shape)
    tiff_out.write(data=tmp_arr, shape=tmp_arr.shape, **metadata.to_dict())


"""