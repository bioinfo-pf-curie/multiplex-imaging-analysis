#!/usr/bin/env python

from instanseg import InstanSeg
import argparse
from pathlib import Path
from tifffile import TiffFile, imwrite

from utils import OmeTifffile



def main(image_path, out_path):
    instanseg_fluo = InstanSeg("fluorescence_nuclei_and_cells", image_reader="bioio", verbosity=1)

    labeled_output = instanseg_fluo.eval(image = image_path,
                                                save_output = True,
                                                save_overlay = True)
    # display = instanseg_brightfield.display(image_tensor, labeled_output)

    metadata = OmeTifffile(TiffFile(image_path).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = labeled_output.dtype

    kwargs = metadata.to_dict(shape=labeled_output.shape)

    imwrite(out_path, labeled_output, bigtiff=True, shaped=False, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help="path for original img")
    parser.add_argument('--out_path', type=str, required=True, help="Output filepath")
    args = parser.parse_args()
    main(image_path=args.img_path, out_path=Path(args.out_path))