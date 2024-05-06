# Outputs

This document describes the output produced by the pipeline. 

## Pipeline overview

The pipeline is built using [Nextflow](https://www.nextflow.io/)
and processes the data using the steps presented in the main README file.  
Briefly, its goal is to segment microscopic images and quantify cells.

The directories listed below will be created in the output directory after the pipeline has finished. 

## Segmentation

**Output directory: `merge_channels`**

* `image_name_merged.tif`
  * correspond to input image with all channels marked in markers.csv file merged into one (Maximum Intensity Projection)

**Output directory: `masks`**

* `image_name_masks.tif`
  * Mask of segmented cells

**Output directory: `outlines`**

* `image_name_outlines.tif`
  * Original image with one channel added which outline cell contour. (can be changed in options to be merged image instead)

## Quantification

**Output directory: `quantification`**

* `image_name_data.csv`
  * csv table to quantifie cells.

## Report

**Output directory: `summary`**

* `trace`
  * directory with reporting files

* `pipelineReport.html`
  * brief overview of the pipeline

* `trace`
  * brief overview of the resulting output
