#!/usr/bin/env nextflow

/*
Copyright Institut Curie 2019-2022
This software is a computer program whose purpose is to analyze high-throughput sequencing data.
You can use, modify and/ or redistribute the software under the terms of license (see the LICENSE file for more details).
The software is distributed in the hope that it will be useful, but "AS IS" WITHOUT ANY WARRANTY OF ANY KIND.
Users are therefore encouraged to test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data. 
The fact that you are presently reading this means that you have had knowledge of the license and that you accept its terms.
*/

/*
========================================================================================
                         
                           __  __ ___   _   
                          |  \/  |_ _| /_\  
                          | |\/| || | / _ \ 
                          |_|  |_|___/_/ \_\
                   
========================================================================================
Analysis Pipeline DSL2 template.
https://patorjk.com/software/taag/
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl=2

// Initialize lintedParams and paramsWithUsage
NFTools.welcome(workflow, params)

// Use lintedParams as default params object
paramsWithUsage = NFTools.readParamsFromJsonSettings("${projectDir}/parameters.settings.json")
params.putAll(NFTools.lint(params, paramsWithUsage))

// Run name
customRunName = NFTools.checkRunName(workflow.runName, params.name)

// Custom functions/variables

/*
===================================
  SET UP CONFIGURATION VARIABLES
===================================
*/

// Stage config files
outputDocsCh = Channel.fromPath("$projectDir/docs/output.md")
outputDocsImagesCh = file("$projectDir/docs/images/", checkIfExists: true)

/*
==========================
 VALIDATE INPUTS
==========================
*/

if (!params.images){
  exit 1, "Missing input image (use --images to indicate image path directory)" 
}
if (!params.markers){
  exit 1, "Missing markers file (use --markers to list markers file path)"
}

/*
===========================
   SUMMARY
===========================
*/

summary = [
  'Pipeline' : workflow.manifest.name ?: null,
  'Version': workflow.manifest.version ?: null,
  'DOI': workflow.manifest.doi ?: null,
  'Run Name': customRunName,
  'Inputs' : params.images ?: null,
  'Max Resources': "${params.maxMemory} memory, ${params.maxCpus} cpus, ${params.maxTime} time per job",
  'Container': workflow.containerEngine && workflow.container ? "${workflow.containerEngine} - ${workflow.container}" : null,
  'Profile' : workflow.profile,
  'OutDir' : params.outDir,
  'WorkDir': workflow.workDir,
  'CommandLine': workflow.commandLine
].findAll{ it.value != null }

workflowSummaryCh = NFTools.summarize(summary, workflow, params)

// Workflows

// Processes
include { getSoftwareVersions } from './nf-modules/common/process/utils/getSoftwareVersions'
include { outputDocumentation } from './nf-modules/common/process/utils/outputDocumentation'
include { mergeChannels } from './nf-modules/common/process/mergeChannels'
include { displayOutline } from './nf-modules/common/process/displayOutline'
include { segmentation } from './nf-modules/common/process/segmentation'
include { quantification } from './nf-modules/common/process/quantification'
include { splitImage } from './nf-modules/common/process/splitImage'
include { stitch } from './nf-modules/common/process/stitch'
include { computeMasks } from './nf-modules/common/process/computeMasks'
include { pyramidize } from './nf-modules/common/process/pyramidize'
include { mergeMasks } from './nf-modules/common/process/mergeMasks'

/*
=====================================
            WORKFLOW 
=====================================
*/


workflow {
  versionsCh = Channel.empty()

  main:

    def tiffPattern = ~/tiff?$/
    def modelList = params.segmentation.models instanceof List ? params.segmentation.models : params.segmentation.models.tokenize(",")
    // Init Channels
    imgCh = Channel.fromPath((params.images =~ tiffPattern) ? params.images : "${params.images}/*.ti{f,ff}")
    imgId = imgCh.map{img -> tuple(NFTools.getImageID(img), img)}
    markersCh = Channel.fromPath(params.markers.endsWith(".csv") ? params.markers : "${params.markers}/*.csv")
    mrkId = markersCh.map{img -> tuple(NFTools.getImageID(img), img)}

    intermediate = markersCh.count().branch{
      solo: it == 1
      multi: it > 1
    }
    inputsOriginal = intermediate.solo.combine(
      imgId.combine(markersCh)
    ).mix(intermediate.multi.combine(
      imgId.join(mrkId)
    )).map{count, name, ipath, mpath -> 
      tuple([
        originalName: name, 
        imagePath: ipath, 
        markersPath: mpath,
        imgSize: ipath.size() as Float
      ], ipath, mpath)}
    
    // subroutines
    outputDocumentation(
      outputDocsCh,
      outputDocsImagesCh
    )

    // PROCESS
    merged = mergeChannels(inputsOriginal)

    splittedImg = splitImage(merged)
    splittedImgResult = splittedImg.transpose().map{nb, meta, splitted -> 
      def newMeta = [
        originalName: meta.originalName, 
        imagePath: meta.imagePath, 
        markersPath: meta.markersPath, 
        nbSplittedFile: nb, 
        splittedName: splitted.name - ~/\.\w+$/, 
        startHeight: NFTools.getStartHeight(splitted),
        imgSize: meta.imgSize
      ] 
      tuple(newMeta, splitted)
    }

    segmented = segmentation(splittedImgResult, modelList)

    groupSegmented = segmented.map{meta, segmentedImg, models ->
      meta['model'] = models
      tuple(groupKey(meta.subMap("originalName", "imagePath", "markersPath", "imgSize", 'model'), meta.nbSplittedFile.toInteger()), meta, segmentedImg)
    }.groupTuple().map{groupedkey, old_meta, segmentedImg -> 
      tuple(groupedkey, segmentedImg)
    }
    flow = stitch(groupSegmented)

    partialMasks = computeMasks(flow)

    partialMaskCh = partialMasks.map{meta, partial ->
      tuple(groupKey(meta.subMap("originalName", "imagePath", "markersPath", "imgSize"), modelList.size()), partial)
    }.groupTuple()

    finalMask = mergeMasks(partialMaskCh)

    outline = displayOutline(finalMask.join(merged))

    pyramidizeCh = Channel.empty()
    .mix(NFTools.setTag(merged, "merge_channels"))
    .mix(NFTools.setTag(outline, "outlines"))
    
    pyramidize(pyramidizeCh)
  
    quant = quantification(finalMask)

    //*******************************************
    // Warnings that will be printed in the mqc report
    warnCh = Channel.empty()
}

workflow.onComplete {
  NFTools.makeReports(workflow, params, summary, customRunName, mqcReport)
}
