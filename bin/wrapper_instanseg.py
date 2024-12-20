#!/usr/bin/env python

from pathlib import Path
import os
from instanseg import InstanSeg
import argparse
from pathlib import Path
import torch
from tifffile import TiffFile, imwrite

from utils import OmeTifffile



def main(image_path, out_path, model_name="fluorescence_nuclei_and_cells", reader="bioio"):
    # take care of model downloading as it is dl for every process otherwise

    model_path = Path(os.environ.get("INSTANSEG_BIOIMAGEIO_PATH")) / model_name / "instanseg.pt"
    if model_path.exists():
        model = torch.jit.load(model_path)  
    else:
        from instanseg.utils.utils import download_model
        # it will be recorded in the path defined in env var : INSTANSEG_BIOIMAGEIO_PATH
        model = download_model(model_name)

    instanseg_fluo = InstanSeg(model, image_reader=reader, verbosity=1)

    labeled_output = instanseg_fluo.eval(image = image_path,
                                         save_overlay = True)
    # display = instanseg_brightfield.display(image_tensor, labeled_output)
    if isinstance(labeled_output, torch.Tensor):
            labeled_output = labeled_output.cpu().detach().numpy()
    labeled_output = labeled_output.astype('uint16').squeeze()

    metadata = OmeTifffile(TiffFile(image_path).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="nuclei_mask")
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