# Nextflow pipeline 
<!-- TODO update with the name of the pipeline -->

[![Nextflow](https://img.shields.io/badge/nextflow-%E2%89%A519.10.0-brightgreen.svg)](https://www.nextflow.io/)
[![Install with](https://anaconda.org/conda-forge/vsc-install/badges/version.svg)](https://conda.anaconda.org/anaconda)
[![Singularity Container available](https://img.shields.io/badge/singularity-available-7E4C74.svg)](https://singularity.lbl.gov/)
[![Docker Container available](https://img.shields.io/badge/docker-available-003399.svg)](https://www.docker.com/)

## Introduction

The pipeline is built using [Nextflow](https://www.nextflow.io), a workflow manager to run tasks across multiple compute infrastructures in a very portable manner.
It supports [conda](https://docs.conda.io) package manager and  [singularity](https://sylabs.io/guides/3.6/user-guide/) / [Docker](https://www.docker.com/) containers making installation easier and results highly reproducible.

## Pipeline summary

<!-- TODO 

Describe here the main steps of the pipeline.

1. Step 1 does...
2. Step 2 does...
3. etc

-->

### Quick help

```bash
nextflow run main.nf --help
N E X T F L O W  ~  version 19.10.0
Launching `main.nf` [stupefied_darwin] - revision: aa905ab621
=======================================================

Usage:

Mandatory arguments:
--images [file]                   Path to input images directory 
--markers [file]                  Path to markers file (one file per image, must be a csv file listing markers name and metadata about it, see docs for more information)

Optionnal arguments:
--overlap [float]                Percentage of overlap between tile (default is 0.1)
--tileHeight [int]               Size in pixel of the height of each tile (default will compute the best height for available memory)
--maskOverlap [int]              Size in pixel of the overlap for computing masks (default is 60 ~ 2 x mean cell size)
--mode [str]                     Normalization used before merging channels. Can be either 'custom', 'hist' or 'no-norm' (default is custom if normalization value are present in markers.csv else it's hist)

Skip options: All are false by default
--skipSoftVersion [bool]         Do not report software version

Other options:
--outDir [dir]                  The output directory where the results will be saved
-w/--work-dir [dir]             The temporary directory where intermediate data will be saved
-name [str]                      Name for the pipeline run. If not specified, Nextflow will automatically generate a random mnemonic

=======================================================
Available profiles
-profile test                    Run the test dataset
-profile conda                   Build a new conda environment before running the pipeline. Use `--condaCacheDir` to define the conda cache path
-profile multiconda              Build a new conda environment per process before running the pipeline. Use `--condaCacheDir` to define the conda cache path
-profile path                    Use the installation path defined for all tools. Use `--globalPath` to define the installation path
-profile multipath               Use the installation paths defined for each tool. Use `--globalPath` to define the installation path
-profile docker                  Use the Docker images for each process
-profile singularity             Use the Singularity images for each process. Use `--singularityImagePath` to define the path of the singularity containers
-profile cluster                 Run the workflow on the cluster, instead of locally

```


### Quick run

The pipeline can be run on any infrastructure from a list of input files as follows:

#### Run the pipeline on a test dataset

See the file `conf/test.config` to set your test image.

```bash
nextflow run main.nf -profile test,conda

```


### Defining the '-profile'

By default (whithout any profile), Nextflow executes the pipeline locally, expecting that all tools are available from your `PATH` environment variable.

In addition, several Nextflow profiles are available that allow:
* the use of [conda](https://docs.conda.io) or containers instead of a local installation,
* the submission of the pipeline on a cluster instead of on a local architecture.

The description of each profile is available on the help message (see above).

Here are a few examples to set the profile options:

#### Run the pipeline locally, using a global environment where all tools are installed (build by conda for instance)
```bash
-profile path --globalPath /my/path/to/bioinformatics/tools
```

#### Run the pipeline on the cluster, using the Singularity containers
```bash
-profile cluster,singularity --singularityImagePath /my/path/to/singularity/containers
```

#### Run the pipeline on the cluster, building a new conda environment
```bash
-profile cluster,conda --condaCacheDir /my/path/to/condaCacheDir

```

For details about the different profiles available, see [Profiles](docs/profiles.md).

### Markers.csv

A marker file is a csv file that provides additional details on the marker of the image.
Here is a simple example:

```
cycle,marker_name,segmentation
0,01_Hoechst,False
1,AF1,
2,04_CD31_Argo515,true
3,05_CD45_Argo555L,True
4,06_CD68_Argo535,True
5,07_Vimentin_Argo550,true
6,08_CD4_Argo572,True
7,09_FOXP3_Argo584,True
8,10_CD8a_Argo602,True

...
```

Cycle indicate the laser number used by orion technology, and segmentation is used to create a merge channels with cyto or membrane channels that can help segmentation.
Other columns can be added but will not be used. 

## Full Documentation

1. [Installation](docs/installation.md)
3. [Running the pipeline](docs/usage.md)
4. [Output and how to interpret the results](docs/output.md)
5. [Troubleshooting](docs/troubleshooting.md)

## Credits

This pipeline has been written by Maxime CORBÉ

## Contacts

For any question, bug or suggestion, please use the issue system or contact the bioinformatics core facility.
