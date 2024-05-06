# Usage

This pipeline is used to segment and quantify large multiplexed tiff image


## Table of contents

* [Introduction](#general-nextflow-info)
* [Running the pipeline](#running-the-pipeline)
* [Main arguments](#main-arguments)
    * [`-profile`](#-profile)
    * [`--images`](#-images)
    * [`--markers`](#-markers)
* [Nextflow profiles](#nextflow-profiles)
* [Job resources](#job-resources)
* [Other command line parameters](#other-command-line-parameters)
    * [`--skip*`](#-skip)
    * [`--outDir`](#-outdir)
    * [`-name`](#-name)
    * [`-resume`](#-resume)
    * [`-c`](#-c)
    * [`--maxMemory`](#-maxmemory)
    * [`--maxTime`](#-maxtime)
    * [`--maxCpus`](#-maxcpus)
* [Profile parameters](#profile-parameters)
    * [`--condaCacheDir`](#-condacachedir)
    * [`--globalPath`](#-globalpath)
    * [`--queue`](#-queue)
    * [`--singularityImagePath`](#-singularityimagepath)

## General Nextflow info

Nextflow handles job submissions on SLURM or other environments, and supervises the job execution. Thus the Nextflow process must run until the pipeline is finished. We recommend that you put the process running in the background through `screen` / `tmux` or similar tool. Alternatively you can run nextflow within a cluster job submitted your job scheduler.

It is recommended to limit the Nextflow Java virtual machines memory. We recommend adding the following line to your environment (typically in `~/.bashrc` or `~./bash_profile`):

```bash
NXF_OPTS='-Xms1g -Xmx4g'
```

## Running the pipeline

The typical command for running the pipeline is as follows:
```bash
nextflow run main.nf --images image.tif --markers markers.csv -profile 'singularity'
```

This will launch the pipeline with the `singularity` configuration profile. See below for more information about profiles.

Note that the pipeline will create the following files in your working directory:

```bash
work            # Directory containing the nextflow working files
results         # Finished results (configurable, see below)
.nextflow_log   # Log file from Nextflow
# Other nextflow hidden files, eg. history of pipeline runs and old logs.
```

You can change the output director using the `--outDir/-w` options.

## Main arguments

### `-profile`

Use this option to set the [Nextflow profiles](profiles.md). For example:

```bash
-profile singularity,cluster
```

### `--images`
Use this to specify the location of your input images files. For example:

```bash
--images 'path/to/image_directory/'
```

### `--markers`
Use this to specify the location of your input markers files. For example:

```bash
--markers 'path/to/markers_directory/'
```

## Nextflow profiles

Different Nextflow profiles can be used. See [Profiles](profiles.md) for details.

## Job resources

Each step in the pipeline has a default set of requirements for number of CPUs, memory and time (see the [`conf/process.conf`](../conf/process.config) file). 
For most of the steps in the pipeline, if the job exits with an error code of `143` (exceeded requested resources) it will automatically resubmit with higher requests (2 x original). If it still fails then the pipeline is stopped.

## Configuration options

  #### `minMemory`

  Minimum ram used by each process. default='2.GB'

  #### `summaryDir`
  
  path for summary directory. default = "${params.outDir}/summary"

  #### `clusterOptions`
  
  Options for cluster (not used in local). default for abacus = " --account dev"


### Masks

  #### `masks.overlap`

  overlap used by dask to compute mask from cellpose's flows. In pixels. default = 120 

### Normalization

  #### `normalization.mode`
  
  Mode used for image normalization before segmentation and before quantification. Accepted values are : ['auto', 'custom', 'hist', 'gaussian', 'equalize']. Default = "auto".

- auto : act as 'custom' if normalization values are given else act as 'hist'

- custom : Perform a normalization based on min/max values given by the user, in markers.csv file (otherwise nothing will be done).
    
- hist : Compute min/max values for normalization based on an histogram of intensities values (100 bins, min = 2 + argmax and max = 90bin)

- gaussian : normalize by applying a gaussian filter.

- equalize : Perform a [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

### Segmentation 
  #### `segmentation.name`

  Change segmenter (only cellpose is correctly implemented as of now). default = "cellpose"

  #### `segmentation.tileHeight`
     
  Fix height tile in image segmentation. An instance of the segmenter will be launched on each tiles. A null value will compute a good value in respect of available memory. Default = null

  #### `segmentation.overlap`
  
  Overlap of the segmenter. Tiles will be further cut into chunk of 224 * 224 pixels (for cellpose by example). In percentage of chunk length. Default = 0.1 (~22 pixels for cellpose)

  #### `segmentation.additionalParms`
  
  A string of additional parameters to add to segmenter command. Default = "--restore_type denoise_cyto3"


### Quantification
    
  #### `quantification.normalization`
  
  Wether or not perform normalization of channels before quantification. Default = true
  

### Outline

  #### `outline` 
  
  Which image will be used for displaying outline. Accepted value are ["merged", "original"]. Default = "merged"

## Other command line parameters

### `--outDir`
The output directory where the results will be saved.

### `-name`
Name for the pipeline run. If not specified, Nextflow will automatically generate a random mnemonic.

This is used in the MultiQC report (if not default) and in the summary HTML.

**NB:** Single hyphen (core Nextflow option)

### `-resume`
Specify this when restarting a pipeline. Nextflow will used cached results from any pipeline steps where the inputs are the same, continuing from where it got to previously.

You can also supply a run name to resume a specific run: `-resume [run-name]`. Use the `nextflow log` command to show previous run names.

**NB:** Single hyphen (core Nextflow option)

### `-c`
Specify the path to a specific config file.

**NB:** Single hyphen (core Nextflow option)

Note - you can use this to override pipeline defaults.

### `--maxMemory`
Use to set a top-limit for the default memory requirement for each process.
Should be a string in the format integer-unit. eg. `--maxMemory '8.GB'`

### `--maxTime`
Use to set a top-limit for the default time requirement for each process.
Should be a string in the format integer-unit. eg. `--maxTime '2.h'`

### `--maxCpus`
Use to set a top-limit for the default CPU requirement for each process.
Should be a string in the format integer-unit. eg. `--maxCpus 1`

## Profile parameters


### `--condaCacheDir`

Whenever you use the `conda` or `multiconda` profiles, the conda environments are created in the ``${HOME}/conda-cache-nextflow`` folder by default. This folder can be changed using the `--condaCacheDir` option.


### `--globalPath`

When you use `path` or `multipath` profiles, the ``path`` and ``multipath`` folders where are installed the tools can be changed at runtime with the ``--globalPath`` option.


### `--queue`

If you want your job to be submitted on a specific ``queue`` when you use the `cluster` profile, use the option ``--queue`` with the name of the queue in the command line.


### `--singularityImagePath`

When you use the `singularity` profile, the path to the singularity containers can be changed at runtime with the ``--singularityImagePath`` option.


