#!/usr/bin/env python

from pathlib import Path
import os
from instanseg import InstanSeg
import argparse
from pathlib import Path
import torch
import numpy as np
from tifffile import TiffFile, imwrite

from utils import OmeTifffile

class CustomReader(InstanSeg):
    def read_image(self, image_str: str):
        """
        Read an image file from disk. (add a custom reader as I can not make the other work)
        :param image_str: The path to the image.
        :return: The image array if it can be safely read (or the path to the image if it cannot) and the pixel size in microns.
        """
        if self.prefered_image_reader == "tiffslide":
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            width,height = slide.dimensions[0], slide.dimensions[1]
            num_pixels = width * height
            if num_pixels < self.medium_image_threshold:
                image_array = slide.read_region((0, 0), 0, (width, height), as_array=True)
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "skimage.io":
            from skimage.io import imread
            image_array = imread(image_str)
            img_pixel_size = None

        elif self.prefered_image_reader == "bioio":
            from bioio import BioImage
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            if num_pixels < self.medium_image_threshold:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size
        elif self.prefered_image_reader == 'custom-ome-tiff':
            from utils import read_tiff_orion
            image_array, mtd = read_tiff_orion(image_str)
            num_pixels = np.cumprod(image_array.shape)[-1]
            img_pixel_size = mtd.pix.physical_size_x
            # if num_pixels < self.medium_image_threshold:
            image_array = np.array(image_array).squeeze()
            # else:
            #    return image_str, img_pixel_size

            
        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented.")
        
        if img_pixel_size is None or float(img_pixel_size) < 0 or float(img_pixel_size) > 2:
            img_pixel_size = self.read_pixel_size(image_str)

        if img_pixel_size is not None:
            import warnings
            if float(img_pixel_size) <= 0 or float(img_pixel_size) > 2:
                warnings.warn(f"Pixel size {img_pixel_size} microns per pixel is invalid.")
                img_pixel_size = None

        return image_array, img_pixel_size
    
    def eval_small_image(self, *args, **kwargs):
        kwargs.pop('overlap', None)
        return super().eval_small_image(*args, **kwargs)


def main(image_path, out_path, model_name="fluorescence_nuclei_and_cells", reader="bioio", only_cells=True):
    # take care of model downloading as it is dl for every process otherwise

    model_path = Path(os.environ.get("INSTANSEG_BIOIMAGEIO_PATH")) / model_name / "instanseg.pt"
    if model_path.exists():
        model = torch.jit.load(model_path)  
    else:
        from instanseg.utils.utils import download_model
        # it will be recorded in the path defined in env var : INSTANSEG_BIOIMAGEIO_PATH
        model = download_model(model_name)

    instanseg_fluo = CustomReader(model, image_reader=reader, verbosity=1)

    labeled_output = instanseg_fluo.eval(image = image_path,
                                         save_overlay = True,
                                         overlap=10) 
    # display = instanseg_brightfield.display(image_tensor, labeled_output)
    if isinstance(labeled_output, torch.Tensor):
            labeled_output = labeled_output.cpu().detach().numpy()
    labeled_output = fastremap.renumber(labeled_output.squeeze()).astype('uint32')

    # maybe remove this when a proper pipeline for both masks is implemented
    if only_cells:
         # only mask for all cell is kept (we get rid of nuclei segmentation, as its not handled by the other jobs, yet...)
         labeled_output = labeled_output[1,...]

    metadata = OmeTifffile(TiffFile(image_path).pages[0])
    metadata.remove_all_channels()
    # metadata.add_channel_metadata(channel_name="nuclei_mask")
    metadata.add_channel_metadata(channel_name="cell_mask")
    metadata.update_shape(labeled_output.shape)

    metadata.dtype = labeled_output.dtype
    
    kwargs = metadata.to_dict()

    imwrite(out_path, labeled_output, bigtiff=True, shaped=False, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help="path for original img")
    parser.add_argument('--out_path', type=str, required=True, help="Output filepath")
    parser.add_argument('--reader', type=str, default="bioio", help="reader used for opening images")
    parser.add_argument('--model_name', type=str, default="fluorescence_nuclei_and_cells", help="name of the model used")
    args = parser.parse_args()
    main(image_path=args.img_path, out_path=Path(args.out_path), model_name=args.model_name, reader=args.reader)