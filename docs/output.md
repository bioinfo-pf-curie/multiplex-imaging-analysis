# Outputs

<!-- TODO update with the output of your pipeline -->

This document describes the output produced by the pipeline. 

## Pipeline overview

The pipeline is built using [Nextflow](https://www.nextflow.io/)
and processes the data using the steps presented in the main README file.  
Briefly, its goal is to process <!-- TODO --> data for any protocol, with or without control samples, and with or without spike-ins.

The directories listed below will be created in the output directory after the pipeline has finished. 

## Segmentation

**Output directory: `merge_channels`**

* `image_merged.tif`
  * correspond to input image with all channels marked in markers.csv file merged into one (Maximum Intensity Projection)

**Output directory: `masks`**

* `image_merged_cp_masks.tif`
  * Mask of segmented cells

**Output directory: `masks`**

* `image_merged_clear_outlines.tif`
  * Compressed RGB image (red=outline, blue=nuclear channel green=resulting of the merge) to be able to visualize segmentation.

## Quantification

**Output directory: `quantification`**

* `image_merged_cp_masks.csv`
  * csv table to quantifie cells.
