#!/usr/bin/env python

# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-applications/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import logging
import sys
import timeit
import argparse

import numpy as np
import tifffile


from deepcell.utils.io_utils import get_image
from deepcell import applications as apps

def load_image(path, channel=0, ndim=3):
    """Load an image file as a single-channel numpy array.

    Args:
        path (str): Filepath to the image file to load.
        channel (list): Loads the given channel if available.
            If channel is list of length > 1, each channel
            will be summed.
        ndim (int): The expected rank of the returned tensor.

    Returns:
        numpy.array: The image channel loaded as an array.
    """
    if not path:
        raise IOError('Invalid path: %s' % path)

    img = get_image(path)

    channel = channel if isinstance(channel, (list, tuple)) else [channel]

    # getting a little tricky, which axis is channel axis?
    if img.ndim == ndim:
        # file includes channels, find the channel axis
        # assuming the channels axis is the smallest dimension
        axis = img.shape.index(min(img.shape))
        if max(channel) >= img.shape[axis]:
            raise ValueError('Channel {} was passed but channel axis is '
                             'only size {}'.format(
                                 max(channel), img.shape[axis]))

        # slice out only the required channel
        slc = [slice(None)] * len(img.shape)
        # use integer to select only the relevant channels
        slc[axis] = channel
        img = img[tuple(slc)]
        # sum on the channel axis
        img = img.sum(axis=axis)

    # expand the (proper) channel axis
    img = np.expand_dims(img, axis=-1)

    if not img.ndim == ndim:
        raise ValueError('Expected image with ndim = {} but found ndim={} '
                         'and shape={}'.format(ndim, img.ndim, img.shape))

    return img


def prepare_mesmer_input(nuclear_path, membrane_path=None, ndim=3,
                         nuclear_channel=0, membrane_channel=0, **kwargs):
    """Load and reshape image input files for the Mesmer application

    Args:
        nuclear_path (str): The path to the nuclear image file
        membrane_path (str): The path to the membrane image file
        ndim (int): Rank of the expected image size
        nuclear_channel (list): Integer or list of integers for the relevant
            nuclear channels of the nuclear image data.
            All channels will be summed into a single tensor.
        membrane_channel (int): Integer or list of integers for the relevant
            nuclear channels of the membrane image data.
            All channels will be summed into a single tensor.

    Returns:
        numpy.array: Single array of input images concatenated on channels.
    """
    # load the input files into numpy arrays
    nuclear_img = load_image(
        nuclear_path,
        channel=nuclear_channel,
        ndim=ndim)

    # membrane image is optional
    if membrane_path:
        membrane_img = load_image(
            membrane_path,
            channel=membrane_channel,
            ndim=ndim)
    else:
        membrane_img = np.zeros(nuclear_img.shape, dtype=nuclear_img.dtype)

    # join the inputs in the correct order
    img = np.concatenate([nuclear_img, membrane_img], axis=-1)

    return img


def validate_input(app, img):
    # validate correct shape of image
    rank = len(app.model_image_shape)
    name = app.__class__.__name__
    errtext = ('Invalid image shape. An image of shape {} was provided, but '
               '{} expects of images of shape [height, widths, {}]'.format(
                   img.shape, str(name).capitalize(), app.required_channels))

    if len(img.shape) != len(app.model_image_shape):
        raise ValueError(errtext)

    if img.shape[rank - 1] != app.required_channels:
        raise ValueError(errtext)


def get_predict_kwargs(kwargs):
    """Returns a dictionary for use in ``app.predict``.

    Args:
        kwargs (dict): Parsed command-line arguments.

    Returns:
        dict: The parsed key-value pairs for ``app.predict``.
    """
    predict_kwargs = dict()
    for k in ['batch_size', 'image_mpp', 'compartment']:
        try:
            predict_kwargs[k] = kwargs[k]
        except KeyError:
            raise KeyError('{} is required for mesmer jobs, but is not found'
                           'in parsed CLI arguments.'.format(k))
    return predict_kwargs


def run_application(arg_dict):
    """Takes the user-supplied command line arguments and runs the specified application

    Args:
        arg_dict: dictionary of command line args

    Raises:
        IOError: If specified output file already exists"""
    _ = timeit.default_timer()

    outfile = os.path.join(arg_dict['output_directory'], arg_dict['output_name'])

    # Check that the output path does not exist already
    if os.path.exists(outfile):
        raise IOError(f'{outfile} already exists!')

    app = apps.Mesmer()

    # load the input image
    image = prepare_mesmer_input(**arg_dict)

    # make sure the input image is compatible with the app
    validate_input(app, image)

    # Applications expect a batch dimension
    image = np.expand_dims(image, axis=0)

    # run the prediction
    kwargs = get_predict_kwargs(arg_dict)
    output = app.predict(image, **kwargs)

    # Optionally squeeze the output
    if arg_dict['squeeze']:
        output = np.squeeze(output)

    # save the output as a tiff
    tifffile.imwrite(outfile, output)

    app.logger.info('Wrote output file %s in %s s.',
                    outfile, timeit.default_timer() - _)
    

def initialize_logger(log_level):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_level = getattr(logging, log_level)

    fmt = '[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s'
    formatter = logging.Formatter(fmt=fmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(log_level)
    logger.addHandler(console)


def get_arg_parser():
    """argument parser to consume command line arguments"""
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a parent parser for common arguments
    # https://stackoverflow.com/a/63283912
    parent = argparse.ArgumentParser()

    class WritableDirectoryAction(argparse.Action):
        # Check that the provided output directory exists, and is writable
        # From: https://gist.github.com/Tatsh/cc7e7217ae21745eb181
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir = values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(
                    '{} is not a valid directory'.format(
                        prospective_dir, ))
            if os.access(prospective_dir, os.W_OK | os.X_OK):
                setattr(namespace, self.dest, os.path.realpath(prospective_dir))
                return
            raise argparse.ArgumentTypeError(
                '{} is not a writable directory'.format(
                    prospective_dir))

    def existing_file(x):
        if x is not None and not os.path.exists(x):
            raise argparse.ArgumentTypeError('{} does not exist.'.format(x))
        return x

    parent.add_argument('--output-directory', '-o',
                        default=os.path.join(root_dir, 'output'),
                        action=WritableDirectoryAction,
                        help='Directory where application outputs are saved.')

    parent.add_argument('--output-name', '-f',
                        default='mask.tif',
                        help='Name of output file.')

    parent.add_argument('-L', '--log-level', default='INFO',
                        choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                        help='Only log the given level and above.')

    parent.add_argument('--squeeze', action='store_true',
                        help='Squeeze the output tensor before saving.')

    # Next, each application is configured as its own subparser
    # Each subparser should inherit from ``parent`` to inherit options
    # Configurable inputs for ``prepare_input`` and ``app.predict`` must
    # match (via name or dest) the function input names exactly.

    # Mesmer Image file inputs
    parent.add_argument('--nuclear-image', '-n', required=True,
                        type=existing_file, dest='nuclear_path',
                        help=('REQUIRED: Path to 2D single channel TIF file.'))

    parent.add_argument('--nuclear-channel', '-nc',
                        default=0, nargs='+', type=int,
                        help='Channel(s) to use of the nuclear image. '
                             'If more than one channel is passed, '
                             'all channels will be summed.')

    parent.add_argument('--membrane-image', '-m',
                        type=existing_file, dest='membrane_path',
                        help=('Path to 2D single channel TIF file. '
                              'Optional. If not provided, membrane '
                              'channel input to network is blank.'))

    parent.add_argument('--membrane-channel', '-mc',
                        default=0, nargs='+', type=int,
                        help='Channel(s) to use of the membrane image. '
                             'If more than one channel is passed, '
                             'all channels will be summed.')

    # Mesmer Inference parameters
    parent.add_argument('--image-mpp', type=float, default=0.5,
                        help='Input image resolution in microns-per-pixel. '
                             'Default value of 0.5 corresponds to a 20x zoom.')

    parent.add_argument('--batch-size', '-b', default=4, type=int,
                        help='Batch size for `model.predict`.')

    parent.add_argument('--compartment', '-c', default='whole-cell',
                        choices=('nuclear', 'whole-cell', 'both'),
                        help='The cellular compartment to segment.')

    return parent


if __name__ == '__main__':

    # get command line args
    ARGS = get_arg_parser().parse_args()

    OUTFILE = os.path.join(ARGS.output_directory, ARGS.output_name)

    initialize_logger(log_level=ARGS.log_level)

    # run application
    run_application(dict(ARGS._get_kwargs()))