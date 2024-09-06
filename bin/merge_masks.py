#!/usr/bin/env python

import argparse
import tifffile
import dask.array as da
import numpy as np
import cv2
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import fastremap
import warnings
# from skimage.segmentation import find_boundaries
import rasterio.features

# import pandas as pd

from dask_utils import correct_edges_inplace, compute_current_cell_id
from utils import OmeTifffile

# optimized from SOPA https://github.com/gustaveroussy/sopa/blob/master/sopa/segmentation/shapes.py

def _ensure_polygon(cell: Polygon | MultiPolygon | GeometryCollection) -> Polygon:
    """Ensures that the provided cell becomes a Polygon

    Args:
        cell: A shapely Polygon or MultiPolygon or GeometryCollection

    Returns:
        The shape as a Polygon
    """
    cell = shapely.make_valid(cell)
    if isinstance(cell, Polygon):
        if cell.interiors:
            cell = Polygon(list(cell.exterior.coords))
        return cell
    if isinstance(cell, MultiPolygon):
        return max(cell.geoms, key=lambda polygon: polygon.area)
    if isinstance(cell, GeometryCollection):
        geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]
        if not geoms:
            print(f"Removing cell of type {type(cell)} as it contains no Polygon geometry")
            return None
        return max(geoms, key=lambda polygon: polygon.area)
    print(f"Removing cell of unknown type {type(cell)}")
    return None


# def _smoothen_cell(cell: MultiPolygon, smooth_radius: float, tolerance: float) -> Polygon | None:
#     """Smoothen a cell polygon

#     Args:
#         cell_id: ID of the cell to geometrize
#         smooth_radius: radius used to smooth the cell polygon
#         tolerance: tolerance used to simplify the cell polygon

#     Returns:
#         Shapely polygon representing the cell, or `None` if the cell was empty after smoothing
#     """
#     cell = cell.buffer(-smooth_radius).buffer(2 * smooth_radius).buffer(-smooth_radius)
#     cell = cell.simplify(tolerance)
#     return None if cell.is_empty else _ensure_polygon(cell)


# def _default_tolerance(mean_radius: float) -> float:
#     if mean_radius < 10:
#         return 0.4
#     if mean_radius < 20:
#         return 1
#     return 2


# def geometrize(
#     mask: np.ndarray, tolerance: float | None = None, smooth_radius_ratio: float = 0.1
# ) -> list[Polygon]:
#     """Convert a cells mask to multiple `shapely` geometries. Inspired from https://github.com/Vizgen/vizgen-postprocessing
    

#     Args:
#         mask: A cell mask. Non-null values correspond to cell ids
#         tolerance: Tolerance parameter used by `shapely` during simplification. By default, define the tolerance automatically.

#     Returns:
#         List of `shapely` polygons representing each cell ID of the mask
#     """
#     max_cells = mask.max()

#     if max_cells == 0:
#         print("No cell was returned by the segmentation")
#         return []
    
#     bounds = pd.DataFrame(np.where(find_boundaries(mask, connectivity=2, mode="inner"), mask, 0))
#     bounds = pd.DataFrame(bounds.unstack())

#     cells = []
#     grouped_bound = bounds.groupby([0], sort=False)
#     for v, idxs in grouped_bound:
#         if v[0] != 0 and len(idxs.index.values) > 4:
#             cells.append(shapely.convex_hull(Polygon(idxs.index.values)))
    
#     mean_radius = np.sqrt(np.array([cell.area for cell in cells]) / np.pi).mean()
#     smooth_radius = mean_radius * smooth_radius_ratio

#     if tolerance is None:
#         tolerance = _default_tolerance(mean_radius)

#     cells = [_smoothen_cell(cell, smooth_radius, tolerance) for cell in cells]
#     cells = [cell for cell in cells if cell is not None]
#     print(
#         f"Percentage of non-geometrized cells: {((grouped_bound.ngroups-1) - len(cells)) / (grouped_bound.ngroups-1):.2%} (usually due to segmentation artefacts)"
#     )

#     return cells


def solve_conflicts(
    cells: list[Polygon],
    threshold: float = 0.5,
    patch_indices: np.ndarray | None = None,
    return_indices: bool = False,
) -> np.ndarray[Polygon] | tuple[np.ndarray[Polygon], np.ndarray]:
    """Resolve segmentation conflicts (i.e. overlap) after running segmentation on patches

    Args:
        cells: List of cell polygons
        threshold: When two cells are overlapping, we look at the area of intersection over the area of the smallest cell. If this value is higher than the `threshold`, the cells are merged
        patch_indices: Patch from which each cell belongs.
        return_indices: If `True`, returns also the cells indices. Merged cells have an index of -1.

    Returns:
        Array of resolved cells polygons. If `return_indices`, it also returns an array of cell indices.
    """
    cells = list(cells)
    n_cells = len(cells)
    resolved_indices = np.arange(n_cells)

    if n_cells == 0:
        warnings.warn("No cells was segmented, cannot resolve conflicts")
        return cells

    tree = shapely.STRtree(cells)

    try:
        conflicts = tree.query(cells, predicate="intersects")
    except:
        return cells

    if patch_indices is not None:
        conflicts = conflicts[:, patch_indices[conflicts[0]] != patch_indices[conflicts[1]]].T
    else:
        conflicts = conflicts[:, conflicts[0] != conflicts[1]].T

    for i1, i2 in conflicts:
        resolved_i1: int = resolved_indices[i1]
        resolved_i2: int = resolved_indices[i2]
        cell1, cell2 = cells[resolved_i1], cells[resolved_i2]

        intersection = cell1.intersection(cell2).area
        if intersection >= threshold * min(cell1.area, cell2.area):
            cell = _ensure_polygon(cell1.union(cell2))

            resolved_indices[np.isin(resolved_indices, [resolved_i1, resolved_i2])] = len(cells)
            cells.append(cell)

    unique_indices = np.unique(resolved_indices)
    unique_cells = np.array(cells)[unique_indices]

    if return_indices:
        return unique_cells, np.where(unique_indices < n_cells, unique_indices, -1)

    return unique_cells


def recreate_mask(cells, shape, idx_start=1):
    """
    From a list of shapes (cells), reconstruct the mask image.

    Parameters
    ----------

    cells: list of Polygon
        list of shape to be draw into the mask
    shape: tuple of int
        image size (same as the size of original masks)
    idx_start: int
        index to start from

    Return
    ------

    result: np.array
        Merged mask
    """
    result = np.zeros(shape=shape)
    for i, cell in enumerate(cells, idx_start):
        result = cv2.fillConvexPoly(result, np.rint(cell.exterior.xy).astype("int32").T, color=i)
    return result


def on_chunk(chunk, threshold, block_info=None, diameter=30):
    """
    Convert each masks into a list of shape (cells), concat these lists, solve intersection and then reconvert it to mask image.

    Parameters
    ----------

    chunk: 3D np.array
        array of number_of_masks * image_width * image_height
    threshold: float
        Intersection over union value for which cells are to be merged

    Return
    ------
     
    merged_mask: 2D np.array
        resulting mask
    """
    cells = []
    for i in range(chunk.shape[0]):
        # cells += geometrize(chunk[i])
        mask = chunk[i].astype('float32')
        for cell in rasterio.features.shapes(mask, mask=mask > 0, connectivity=8):
            if len(cell[0]['coordinates'][0]) > 4:
                polygon = Polygon(cell[0]['coordinates'][0]).buffer(0)
                if not polygon.is_valid:
                    polygon = shapely.simplify(polygon, preserve_topology=False)
                if not polygon.is_valid:
                    print(polygon)
                    print('not_valid\n')
                if not polygon.is_closed:
                    print(polygon)
                    print("not closed\n")
                if polygon.is_ring:
                    print(polygon)
                    print('is ring\n')
                cells.append(polygon)

    results = solve_conflicts(cells, threshold=threshold)

    current_cell_id = compute_current_cell_id(block_info, mean_cell_area=np.pi * (diameter / 2) ** 2)

    return recreate_mask(results, chunk.shape[1:], current_cell_id)


def merge_masks(list_of_masks, chunk_size=1024, overlap=120, threshold=0.5, diameter=30):
    """
    Merge a list of masks (cells labels images) into one, based on a threshold of percentage of intersection
    (see SOPA for a more detailed implementation of solve conflict)

    Parameters
    ----------

    list_of_masks: List of (string | Path)
        List of path of the masks to be merged
    chunk_size: int
        size of each chunk (square of chunk_size by chunk_size)
    overlap: int
        number of pixels taken for the previous and next chunk
    threshold: float
        Intersection over union value for which cells are to be merged

    Return
    ------

    None

    """
    masks = [da.from_zarr(tifffile.TiffFile(mask).series[0].aszarr(), chunks=(chunk_size, chunk_size)) for mask in list_of_masks]
    masks = da.stack(masks)
    final_mask = da.map_overlap(on_chunk, masks, dtype=np.uint32, depth={0: 0, 1: overlap, 2: overlap}, drop_axis=0, threshold=threshold, diameter=diameter).compute()

    correct_edges_inplace(final_mask, chunks_size=(chunk_size, chunk_size))

    fastremap.renumber(final_mask, in_place=True) #convenient to guarantee non-skipped labels

    return final_mask.astype('uint32')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_of_mask', nargs="+", type=str, required=True, help="filename for masks to merge")
    parser.add_argument('--out', type=str, required=True, help="Output path for resulting image")
    parser.add_argument('--overlap', type=int, default=120, required=False, help="overlap between tiles")
    parser.add_argument('--chunk_size', type=int, default=8192, required=False, help="size of tiles")
    parser.add_argument('--threshold', type=float, default=0.5, required=False, help="Intersection over union for cells to be merged")
    parser.add_argument('--original', type=str, required=False, help="path to original image (metadata except dtype and channels info will be copied)")
    parser.add_argument('--diameter', type=float, default=30, required=False, help="mean diameter (in pixels) of cells")
    args = parser.parse_args()
    # merge_masks_wo_dask(args.list_of_mask, args.out, threshold=args.threshold)
    mask = merge_masks(args.list_of_mask, overlap=args.overlap, chunk_size=args.chunk_size, threshold=args.threshold, diameter=args.diameter)

    kwargs = {}
    if args.original:
        metadata = OmeTifffile(tifffile.TiffFile(args.original).pages[0])
        metadata.remove_all_channels()
        metadata.add_channel_metadata(channel_name="masks")
        metadata.dtype = mask.dtype
        kwargs.update(metadata.to_dict(shape=mask.shape))

    tifffile.imwrite(args.out, mask, bigtiff=True, shaped=False, **kwargs)
