#!/usr/bin/env python

import argparse

from tifffile import TiffFile, imwrite

import warnings

import numpy as np
import scipy.ndimage as nd

from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects, h_maxima
from skimage.morphology import disk, ball, square, cube, dilation
from skimage.segmentation import relabel_sequential, watershed

from utils import OmeTifffile


def deep_watershed(outputs,
                   radius=10,
                   maxima_threshold=0.1,
                   interior_threshold=0.01,
                   maxima_smooth=0,
                   interior_smooth=1,
                   maxima_index=0,
                   interior_index=-1,
                   small_objects_threshold=0,
                   pixel_expansion=None,
                   maxima_algorithm='h_maxima',
                   **kwargs):
    """ From https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/deep_watershed.py
    Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    Args:
        outputs (list): List of [maximas, interiors] model outputs.
            Use `maxima_index` and `interior_index` if list is longer than 2,
            or if the outputs are in a different order.
        radius (int): Radius of disk used to search for maxima
        maxima_threshold (float): Threshold for the maxima prediction.
        interior_threshold (float): Threshold for the interior prediction.
        maxima_smooth (int): smoothing factor to apply to ``maximas``.
            Use ``0`` for no smoothing.
        interior_smooth (int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
        maxima_index (int): The index of the maxima prediction in ``outputs``.
        interior_index (int): The index of the interior prediction in
            ``outputs``.
        small_objects_threshold (int): Removes objects smaller than this size.
        pixel_expansion (int): Number of pixels to expand ``interiors``.
        maxima_algorithm (str): Algorithm used to locate peaks in ``maximas``.
            One of ``h_maxima`` (default) or ``peak_local_max``.
            ``peak_local_max`` is much faster but seems to underperform when
            given regious of ambiguous maxima.

    Returns:
        numpy.array: Integer label mask for instance segmentation.

    Raises:
        ValueError: ``outputs`` is not properly formatted.
    """
    try:
        maximas = outputs[maxima_index]
        interiors = outputs[interior_index]
    except (TypeError, KeyError, IndexError):
        raise ValueError('`outputs` should be a list of at least two '
                         'NumPy arryas of equal shape.')

    valid_algos = {'h_maxima', 'peak_local_max'}
    if maxima_algorithm not in valid_algos:
        raise ValueError('Invalid value for maxima_algorithm: {}. '
                         'Must be one of {}'.format(
                             maxima_algorithm, valid_algos))

    total_pixels = maximas.shape[1] * maximas.shape[2]
    if maxima_algorithm == 'h_maxima' and total_pixels > 5000**2:
        warnings.warn('h_maxima peak finding algorithm was selected, '
                      'but the provided image is larger than 5k x 5k pixels.'
                      'This will lead to slow prediction performance.')
    # Handle deprecated arguments
    min_distance = kwargs.pop('min_distance', None)
    if min_distance is not None:
        radius = min_distance
        warnings.warn('`min_distance` is now deprecated in favor of `radius`. '
                      'The value passed for `radius` will be used.',
                      DeprecationWarning)

    # distance_threshold vs interior_threshold
    distance_threshold = kwargs.pop('distance_threshold', None)
    if distance_threshold is not None:
        interior_threshold = distance_threshold
        warnings.warn('`distance_threshold` is now deprecated in favor of '
                      '`interior_threshold`. The value passed for '
                      '`distance_threshold` will be used.',
                      DeprecationWarning)

    # detection_threshold vs maxima_threshold
    detection_threshold = kwargs.pop('detection_threshold', None)
    if detection_threshold is not None:
        maxima_threshold = detection_threshold
        warnings.warn('`detection_threshold` is now deprecated in favor of '
                      '`maxima_threshold`. The value passed for '
                      '`detection_threshold` will be used.',
                      DeprecationWarning)

    if maximas.shape[:-1] != interiors.shape[:-1]:
        raise ValueError('All input arrays must have the same shape. '
                         'Got {} and {}'.format(
                             maximas.shape, interiors.shape))

    if maximas.ndim not in {4, 5}:
        raise ValueError('maxima and interior tensors must be rank 4 or 5. '
                         'Rank 4 is 2D data of shape (batch, x, y, c). '
                         'Rank 5 is 3D data of shape (batch, frames, x, y, c).')

    input_is_3d = maximas.ndim > 4

    label_images = []
    for maxima, interior in zip(maximas, interiors):
        # squeeze out the channel dimension if passed
        maxima = nd.gaussian_filter(maxima[..., 0], maxima_smooth)
        interior = nd.gaussian_filter(interior[..., 0], interior_smooth)

        if pixel_expansion:
            fn = cube if input_is_3d else square
            interior = dilation(interior, footprint=fn(pixel_expansion * 2 + 1))

        # peak_local_max is much faster but has poorer performance
        # when dealing with more ambiguous local maxima
        if maxima_algorithm == 'peak_local_max':
            coords = peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get('exclude_border', False))

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1
        else:
            # Find peaks and merge equal regions
            fn = ball if input_is_3d else disk
            markers = h_maxima(image=maxima,
                               h=maxima_threshold,
                               footprint=fn(radius))

        markers = label(markers)
        label_image = watershed(-1 * interior, markers,
                                mask=interior > interior_threshold,
                                watershed_line=0)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(label_image,
                                               min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images


def deep_watershed_mibi(model_output,
                        interior_model='pixelwise-interior',
                        maxima_model='inner-distance',
                        **kwargs):
    """DEPRECATED. Please use ``deep_watershed`` instead.

    Postprocessing function for multiplexed deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the pixelwise interior prediction.

    Args:
        model_output (dict): DeepWatershed model output. A dictionary containing key: value pairs
            with the transform name and the corresponding output. Currently supported keys:

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Foreground prediction for the foregound/background transform.
            - pixelwise_interior: Interior prediction for the interior/border/background transform.

        interior_model (str): Name of semantic head used to predict interior
            of each object.
        maxima_model (str): Name of semantic head used to predict maxima of
            each object.
        kwargs (dict): Keyword arguments for ``deep_watershed``.

    Returns:
        numpy.array: Uniquely labeled mask.

    Raises:
        ValueError: if ``interior_model`` or ``maxima_model`` is invalid.
        ValueError: if ``interior_model`` or ``maxima_model`` predictions
            do not have length 4
    """
    text = ('deep_watershed_mibi is deprecated and will be removed in a '
            'future version. Please use '
            '`deepcell_toolbox.deep_watershed.deep_watershed` instead.')
    warnings.warn(text, DeprecationWarning)

    interior_model = str(interior_model).lower()
    maxima_model = str(maxima_model).lower()

    valid_model_names = {'inner-distance', 'outer-distance',
                         'fgbg-fg', 'pixelwise-interior'}

    zipped = zip(['interior_model', 'maxima_model'],
                 [interior_model, maxima_model])

    for name, model in zipped:
        if model not in valid_model_names:
            raise ValueError('{} must be one of {}, got {}'.format(
                name, valid_model_names, model))

        arr = model_output[model]
        if len(arr.shape) != 4:
            raise ValueError('Model output must be of length 4. The {} {} '
                             'output provided is of shape {}.'.format(
                                 name, model, arr.shape))

    output = [model_output[maxima_model], model_output[interior_model]]

    label_images = deep_watershed(output, **kwargs)

    return label_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True, help="filename of mesmer output in tiff format")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--original', type=str, required=True, help="File path of original image (to get metadata from)")
    parser.add_argument('--chunks', type=int, nargs=2, required=False, default=(1024, 1024), help="Size of chunk for dask")
    parser.add_argument('--overlap', type=int, required=False, default=60, help="Overlap (in pixel) for dask to perform computing of masks on chunks")
    args = parser.parse_args()

    mesmer_output = TiffFile(vars(args)['in'])
    label_img = deep_watershed(mesmer_output)

    metadata = OmeTifffile(TiffFile(args.original).pages[0])
    metadata.remove_all_channels()
    metadata.add_channel_metadata(channel_name="masks")

    metadata.dtype = label_img.dtype

    imwrite(args.out, label_img, bigtiff=True, shaped=False)
