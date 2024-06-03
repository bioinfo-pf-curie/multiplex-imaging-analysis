include { segmentation } from '../process/segmentation'
include { splitImage } from '../process/splitImage'
include { stitch } from '../process/stitch'
include { computeMasks } from '../process/computeMasks'

workflow cellposeSegmentation {
    take:
      image
      modelList

    main:

    splittedImg = splitImage(image)
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

    mask = computeMasks(flow)

    emit:
      mask
}

workflow {
    def modelList = params.modelList ?: ['tissuenet_cp3']
    params.segmentation = [namessh mc: 'cellpose']

    imageCh = Channel.fromPath(params.image)

    cellposeSegmentation(imageCh, modelList)
}