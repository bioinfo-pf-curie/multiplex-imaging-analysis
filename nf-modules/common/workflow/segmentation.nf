include { splitImage } from '../process/splitImage'
include { stitch } from '../process/stitch'
include { computeMasks } from '../process/computeMasks'
include { mergeMasks } from '../process/mergeMasks'

process seg {
  label "${params.segmentation.name}"
  label 'onlyLinux' // only for geniac lint...


  memory {params.segmentation.memory}
  cpus {params.segmentation.cpu}

  input:
    tuple val(meta), path(image)
    each models

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff'), val(models)

  when:
  task.ext.when == null || task.ext.when

  script:
    def cellpose = "cellpose --channel_axis 0 --savedir . --chan 2 --chan2 1 --image_path $image --pretrained_model $models $params.cellpose.additionalParms"
    def mesmer = "wrapper_mesmer.py --squeeze --output-directory . --output-name ${meta.splittedName}_masks.tiff --nuclear-image $image --membrane-image $image --membrane-channel 1"
    def script = params.segmentation.name == 'cellpose' ? cellpose : mesmer
    """
    $script
    """
}


workflow segmentation {
    take:
      metaAndImagesCh
      modelList

    main:

      splittedImg = splitImage(metaAndImagesCh)
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

      segmented = seg(splittedImgResult, modelList)

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
      }.groupTuple().branch{
        solo : modelList.size() == 1
        multiple : true
      }

      finalMask = mergeMasks(partialMaskCh.multiple).mix(partialMaskCh.solo)

    emit:
      finalMask
}