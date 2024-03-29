import tifffile
import zarr
import os
from ome_types import OME, model
import copy
import warnings
import xml.etree.ElementTree as ET
import numpy as np

def get_current_height(npy_path):
    """Helper to parse filename to get position in height for the corresponding tile"""
    npy_name = os.path.basename(npy_path)
    while True:
        npy_name, height = npy_name.rsplit('_', 1)
        try:
            return int(height)
        except ValueError:
            pass
        if not npy_name:
            raise ValueError(f'Height of image {npy_name} not found')

def _tile_generator(arr, channel, x, y, chunk_x, chunk_y):
    """Generate chunk of arr"""
    for x_cur in range(0, x, chunk_x):
        for y_cur in range(0, y, chunk_y):
            yield arr[channel, x_cur: x_cur + chunk_x, y_cur: y_cur + chunk_y]


def min_max_norm(a, min_, max_):
    a = (a - min_) / (max_ - min_)
    a = np.clip(a, 0, 1)
    return a * (2**16 - 1)

def parse_normalization_values(df):
    normalization_col_name = "normalization"
    if normalization_col_name in df.columns:
        return df[normalization_col_name].str.split(";", expand=True).fillna({0:0, 1:2**16}).astype(int).reset_index(drop=True).T.to_dict('list')
    
def compute_hist(img, channel, x, y, chunk_x, chunk_y, img_min=None, img_max=None, num_bin=100, max_bin=0.9, s_factor=2):
    """
    Compute the histogram of a channel from an image and get automatically min and max index for normalizing image afterward.
    The method used to get those are more or less the one used to get it manually. 
    'a little' after the pic (background) and remove 10% extremum for max
    
    Parameters
    ----------

    img: np.array
        image to analyze
    channel: int
        index of the channel of interest
    x: int
        height of img
    y: int
        witdh of img
    chunk_x: int
        the size in x of the tile to compute hist from
    chunk_y: int
        the size in y of the tile to compute hist from
    img_min: int
        minimal intensity value from img (can not be computed easily without img in full memory)
    img_max: int
        maximal intensity value from img
    num_bin: int
        number of bin for histogram (more = more precise, less = efficient)    
    max_bin: float
        max percentage to keep
    s_factor: int
        represent how much of the background we want to clip. Result in a loss of information in low values

    Return
    ------

    tuple of int
        value of the bin to normalize img.
    """
    if img_min is None or img_max is None:
        img_min, img_max = 0, 65535
    bins = np.linspace(img_min, img_max, num_bin)
    hist = np.zeros(num_bin-1)
    for tile in _tile_generator(img, channel, x, y, chunk_x, chunk_y):
        hist += np.histogram(tile, bins)[0]

    idx_min = hist.argmax() + s_factor
    idx_max = int(num_bin * max_bin)
    idx_max = idx_max if len(hist) > idx_max > idx_min else idx_min + 1

    if idx_min >= len(hist)-2:
        res = bins[-2], bins[-1]
    else:
        res = bins[idx_min], bins[idx_max]

    return res

def get_info_qptiff(qptiff):
    # PX = 0.325, PXU = µm, PY = 0.325, PYU = µm, PZ = 1, PZU=µm, size_c, size_t=1, size_z=1, size_x, size_y, dtype=uint16

    result = dict(
        PXU = "µm", PYU = "µm", PZU = "µm",
        PZ = 1, size_t=1, size_z=1,
        dtype="uint16"
    )

    root = ET.fromstring(qptiff).find("ScanProfile")[0] # "ExperimentV4"

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


def make_ome_data(size_x, size_y, size_c, dtype="uint16", **kwargs):
    nominal_magnification = kwargs.pop("nominal_magnification", 20.0)
    dimension_order = kwargs.pop("dimension_order", "XYZCT")
    size_t = kwargs.pop('size_t', 1)
    size_z = kwargs.pop('size_z', 1)

    physical_size_x = kwargs.pop("physical_size_x", 0.325)
    physical_size_y = kwargs.pop("physical_size_y", 0.325)
    physical_size_z = kwargs.pop("physical_size_z", 1.0)
    physical_size_x_unit = kwargs.pop("physical_size_x_unit", 'µm')
    physical_size_y_unit = kwargs.pop("physical_size_y_unit", 'µm')
    physical_size_z_unit = kwargs.pop("physical_size_z_unit", 'µm')

    return OME(
        instruments=[model.Instrument(
            id="Instrument:0", 
            objectives=[model.Objective(id="Objective:0", nominal_magnification=nominal_magnification)]
        )], # result for Orion
        images=[model.Image(
            id="Image:0", instrument_ref={'id': "Instrument:0"}, objective_settings={'id': "Objective:0"},
            pixels=model.Pixels(
                id="Pixels:0", dimension_order=dimension_order, type=dtype,
                big_endian="false", 
                size_x=size_x, size_y=size_y, size_z=size_z, size_c=size_c, size_t=size_t, 
                physical_size_x=physical_size_x, physical_size_x_unit=physical_size_x_unit, 
                physical_size_y=physical_size_y, physical_size_y_unit=physical_size_y_unit,
                physical_size_z=physical_size_z, physical_size_z_unit=physical_size_z_unit,
                tiff_data_blocks=kwargs.pop("tiff_data_blocks", []),
                channels=kwargs.pop('channels', []),
                planes=kwargs.pop("planes", [])
            )
        )],
        structured_annotations=kwargs.pop('structured_annotations', [])
    )


def read_tiff_orion(img_path, idx_serie=0, idx_level=0, *args, **kwargs):
    """
    Helper to read a ome tiff and get its metadata
    
    Parameters
    ----------

    img_path: Path or str
        path of the ome tiff
    idx_serie: int
        index of the serie in tiff (a tiff format is a container and as such can hold multiple images). Not used in Orion ome tiff.
    idx_level: int
        index of the resolution (lower is better resolution, 0 is full) for pyramidal images
    args: list of any
        positional args to be passed to tifffile.TiffFile
    kwargs: dict of any
        keyword args to be passed to tifffile.TiffFile

    Return
    ------

    img: Zarr array
        return image in array format from zarr (lazily loaded)

    metadata: OmeTifffile
        metadata from ome tiff arranged in a pythonnic way (see OmeTifffile)
    """
    tiff = tifffile.TiffFile(img_path, *args, **kwargs)
    zarray = zarr.open(tiff.series[idx_serie].aszarr())
    return (zarray[idx_level] if idx_level is not None and tiff.series[idx_level].is_pyramidal else zarray), OmeTifffile(tiff.pages[0])

class OmeTifffile(object):
    """
    Create a storage for metadata from ome tiff file
    It will read the imageDescription tag (among the others) to get ome tiff metadata based on OME library 
    (and https://docs.openmicroscopy.org/ome-model/5.5.7/ome-tiff/specification.html)
    That can be updated when manipulating images using methods from this class.
    """
    direct_props = {'PhotometricInterpretation': "photometric",  
                    "PlanarConfiguration": "planarconfig", 
                    'Compression': "compress", 
                    "Software": "software"}

    def __init__(self, tifffile_metadata, **kwargs):
        self.tags = {"resolution": [None, None, None], "extratags": []}
        self.ome = None
        self.size = [None, None]
        qptiff_xml = None

        for tag in tifffile_metadata.tags:
            if tag.name == "ImageDescription":
                try:
                    self.ome = OME.from_xml(tag.value, **kwargs)
                except ValueError:
                    # qptiff format (from CODEX, WIP)
                    qptiff_xml = tag.value

            elif tag.name in self.direct_props.keys():
                try:
                    self.tags[self.direct_props[tag.name]] = tag.value.value
                except AttributeError:
                    self.tags[self.direct_props[tag.name]] = tag.value

            elif tag.name in ("XResolution", "YResolution", "ResolutionUnit"):
                self.tags["resolution"][("XResolution", "YResolution", "ResolutionUnit").index(tag.name)] = tag.value

            else:
                self.tags['extratags'].append((tag.code, tag.dtype, tag.count, tag.value, True)) # true for tag.writeonce (orion is one image per tiff)
                if tag.code == 256:
                    self.size[0] = tag.value
                if tag.code == 257:
                    self.size[1] = tag.value

        self.dtype = tifffile_metadata.dtype

        if self.ome is None:
            default = get_info_qptiff(qptiff_xml) if qptiff_xml is not None else {}
            default['size_x'] = self.size[1]
            default['size_y'] = self.size[0]
            default['dtype'] = self.dtype
            default.update(kwargs)

            try:
                self.ome = make_ome_data(**default)
            except BaseException:
                self.ome = make_ome_data(1,1,1) # better default ? i don't want to fail when there is 0 metadata
            
        if self.tags.get('planarconfig', None) == 1 and kwargs.get('force_planarconfig', True):
            warnings.warn("Planar Configuration read as 1 (contigue) will be removed from metadata."
                          "Orion used to not correctly set this to 1. To keep the same planarconfig set force_planarconfig to False")
            self.tags.pop('planarconfig')
            

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
        """update the shape of image"""
        for idx, char in enumerate(order):
            self.pix.__setattr__(f"size_{char.lower()}", arr_shape[idx])

    def to_dict(self, dtype=True, shape=None):
        """transform this class to a dict of parameters, each of them can be passed to tifffile.write and assimilated"""
        this_dict = self.tags.copy()
        this_dict['compression'] = this_dict.pop('compress')

        if shape is not None:
            self.pix.size_y=shape[0]
            self.pix.size_x=shape[1]

        elif self.pix.size_x == 1 or self.pix.size_y == 1:
            raise ValueError(f"About to write an image with shape (x={self.pix.size_x}, y={self.pix.size_y})." 
                             "Probably because of wrong instanciation. To remove error, please add shape in to_dict params.")

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
        """Allow to add an existing channel from model.Channel into self 
        (use add_channel_metadata when you want to create a new one)
        
        Parameters
        ----------

        channel_data: model.Channel
            channel to be added
        """
        # sometimes planes are not registered here
        if len(self.pix.planes) == len(self.pix.channels):
            the_c = 0 if self.pix.size_c == 1 and len(self.pix.planes) == 0 else int(self.pix.size_c)
            # particular case due to validation error on size_c if = 0
            self.pix.planes.append(model.Plane(the_z=0, the_t=0, the_c=the_c))
        try:
            self.pix.tiff_data_blocks = [{'uuid': None, 'ifd': 0, 'first_z': 0, 'first_t': 0, 'first_c': 0, 'plane_count': len(self.pix.planes)}]
        except BaseException as e: 
            print("add channel error")
            print(e)
        self.pix.channels.append(channel_data)
        self.pix.size_c = len(self.pix.channels)
    
    def add_channel_metadata(self, channel_name, add_prefix=True, **kwargs):
        """
        Create a new channel object with name = channel_name and add it to the list.
        
        Parameters
        ----------
        
        channel_name: str
            name of channel
        
        add_prefix: bool
            if true, it will add a number as a prefix of the channel name (respecting our convention)

        kwargs: dict of any
            keyword args to be passed at model.Channel when creating it.
        """
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
            self.pix.tiff_data_blocks = []
        except BaseException as e: 
            print(e)

    def get_channel(self, id_chan):
        return self.pix.channels[id_chan]
    
    def copy(self):
        return copy.deepcopy(self)
        
"""
# sample code to use this (it was used to make small images as test set)

img_path = "orion/data/2017206/220516_Lung_18p_P39_A28_C76dX_E16_Curie18@20220519_110224_090101.ome.tiff"
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

with TiffWriter("image_test.ome.tiff", bigtiff=True, shaped=False) as tiff_out:
    tmp_arr = img[:, 5079:5150, 5698:5754]
    metadata.update_shape(tmp_arr.shape)
    tiff_out.write(data=tmp_arr, shape=tmp_arr.shape, **metadata.to_dict())


"""