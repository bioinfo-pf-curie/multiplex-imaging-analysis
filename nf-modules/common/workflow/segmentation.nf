include { splitImage } from '../process/splitImage'
include { stitch } from '../process/stitch'
include { computeMasks } from '../process/computeMasks'
include { mergeMasks } from '../process/mergeMasks'

process seg {
  label "${params.segmentation.name}"
  label 'infiniteTime'
  label 'onlyLinux' // only for geniac lint...

  input:
    tuple val(meta), path(image)
    each models

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff'), val(models), stdout

  when:
  task.ext.when == null || task.ext.when

  script:
    def cellpose = "cellpose --channel_axis 0 --savedir . --diameter $params.segmentation.diameter --chan 2 --chan2 1 --image_path $image --pretrained_model $models $params.cellpose.additionalParms"
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

      seg(splittedImgResult, modelList)

      groupSegmented = seg.out[0].map{meta, segmentedImg, models, diameter ->
        meta['model'] = models
        meta['diameter'] = (diameter.toString() =~ /using diameter (\d+\.?\d*)/)
        if (meta['diameter']) {
          meta['diameter'] = meta['diameter'][0][1] as Float
        } else {
          meta['diameter'] = null
        }
        tuple(groupKey(meta.subMap("originalName", "imagePath", "markersPath", "imgSize", 'model', 'diameter'), meta.nbSplittedFile.toInteger()), meta, segmentedImg)
      }.groupTuple().map{groupedkey, old_meta, segmentedImg -> 
        tuple(groupedkey, segmentedImg)
      }
      flow = stitch(groupSegmented)

      partialMasks = computeMasks(flow)

      partialMaskCh = partialMasks.map{meta, partial ->
        tuple(groupKey(meta.subMap("originalName", "imagePath", "markersPath", "imgSize"), modelList.size()), partial, meta["diameter"])
      }.groupTuple().branch{
        solo : modelList.size() == 1
        multiple : true
      }

      finalMask = mergeMasks(partialMaskCh.multiple).mix(partialMaskCh.solo.map{meta, mask, diam ->
        tuple(meta, mask)
      })

    emit:
      finalMask
}