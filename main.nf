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
// mqcReport = []
// include {checkAlignmentPercent} from './lib/functions'

/*
===================================
  SET UP CONFIGURATION VARIABLES
===================================
*/

// Initialize variable from the genome.conf file
//params.bowtie2Index = NFTools.getGenomeAttribute(params, 'bowtie2')

// Stage config files
//multiqcConfigCh = Channel.fromPath(params.multiqcConfig)
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
==========================
 BUILD CHANNELS
==========================
*/

// if ( params.metadata ){
//   Channel
//     .fromPath( params.metadata )
//     .ifEmpty { exit 1, "Metadata file not found: ${params.metadata}" }
//     .set { metadataCh }
// }




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
include { mergeChannels } from './nf-modules/common/process/merge_channels'
include { displayOutline } from './nf-modules/common/process/display_outline'
include { segmentation } from './nf-modules/common/process/segmentation'
include { quantification } from './nf-modules/common/process/quantification'
include { splitImage } from './nf-modules/common/process/split_image'
include { mergeSegmentation } from './nf-modules/common/process/merge_segmentation'
include { pyramidize } from './nf-modules/common/process/pyramidize'

process formInputs {
    input: 
      val count
      val imgs
      val markers

    output: val out
    
    script:
      imgsCh = imgs.collect{img -> tuple(NFTools.getImageID(img), img)}
      out = (count == 1) ? imgsCh.collect {img -> tuple(*img, markers[0])} : [imgsCh, markers.collect{mrk -> tuple(NFTools.getImageID(mrk), mrk)}].groupBy{it.head()}
      """echo $out"""
}

/*
=====================================
            WORKFLOW 
=====================================
*/


workflow {
  versionsCh = Channel.empty()

  main:
    // Init Channels
    imgCh = Channel.fromPath([params.images, "${params.images}/*.tiff"])
    markersCh = Channel.fromPath(params.markers.endsWith(".csv") ? params.markers : "${params.markers}/*.csv")
    
    inputs_original = formInputs(markersCh.count(), imgCh.collect(), markersCh.collect()).map(it -> tuple(*it))
    
    // subroutines
    outputDocumentation(
      outputDocsCh,
      outputDocsImagesCh
    )

    // PROCESS
    mergedCh = mergeChannels(inputs_original, ['custom'])

    splitedImg = splitImage(mergedCh)
    splitedImgCh = splitedImg.transpose().map{
      original_name, splitted, original_path -> tuple(original_name, NFTools.getImageID(splitted), splitted, original_path)
    }

    segmented = segmentation(splitedImgCh)
    segmCh = segmented.groupTuple().combine(inputs_original, by:0)

    plainSegm = mergeSegmentation(segmCh)
    plainSegmCh = plainSegm.combine(mergedCh, by:0)
    
    outline = displayOutline(plainSegmCh)

    pyramidizeInput = Channel.empty()
    .mix(NFTools.setTag(mergedCh, "merge_channels"))
    .mix(NFTools.setTag(outline, "outlines"))
    
    pyramidize(pyramidizeInput)
  
    quant = quantification(plainSegmCh)

    //*******************************************
    // Warnings that will be printed in the mqc report
    warnCh = Channel.empty()
}

workflow.onComplete {
  NFTools.makeReports(workflow, params, summary, customRunName, mqcReport)
}
