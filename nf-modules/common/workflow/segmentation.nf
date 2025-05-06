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
    each model
    val segmenterConfig

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff'), val(model), stdout

  when:
  task.ext.when == null || task.ext.when

  script:
    def customParms = ""
    if (segmenterConfig.containsKey('model')) {
      customParms += "$segmenterConfig.model $model"
    }
    if (segmenterConfig.containsKey('membrane-input')) {
      customParms += "$segmenterConfig.membraneInput $image"
    }
    if (segmenterConfig.containsKey('output')) {
      customParms += "$segmenterConfig.output \"${meta.splittedName}_masks.tiff\""
    }
    """
    export diameter="$segmenterConfig.diameter" # there is maybe a better way....
    export INSTANSEG_BIOIMAGEIO_PATH="${params.condaCacheDir}/bioimageio_models/" 
    $segmenterConfig.cmd $segmenterConfig.input $image $segmenterConfig.baseParms $customParms $segmenterConfig.additionalParms
    """
}


workflow segmentation {
    take:
      metaAndImagesCh

    main:
      def segmenterConfig = new File(params.segmentation.config).withReader{
        reader -> new ConfigSlurper().parse(reader.text)[params.segmentation.name]
      }      
      
      // Update segmenterConfig with values from params.segmentation
      params.segmentation.each { key, value ->
        if (!(key in segmenterConfig) || value) {
          segmenterConfig[key] = value
        }
      }
      splittedImg = splitImage(metaAndImagesCh, segmenterConfig)
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

      def modelList = segmenterConfig.modelsList
      modelList = modelList instanceof List ? modelList : modelList.tokenize(",")

      seg(splittedImgResult, modelList, segmenterConfig)

      groupSegmented = seg.out[0].map{meta, segmentedImg, models, diameter ->
        meta['model'] = models
        // get diameter from cellpose output
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
      flow = stitch(groupSegmented, segmenterConfig).map{
        meta, fl -> 
        meta.put('flowSize', fl.size() as Float)
        tuple(meta, fl)
      } // imgSize can not be trusted because input image can (and should) be compressed
      
      partialMasks = computeMasks(flow, segmenterConfig)

      partialMaskCh = partialMasks.map{meta, partial ->
        tuple(groupKey(meta.subMap("originalName", "imagePath", "markersPath", "imgSize", 'flowSize'), modelList.size()), partial, meta["diameter"])
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