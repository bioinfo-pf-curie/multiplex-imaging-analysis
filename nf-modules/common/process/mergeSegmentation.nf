process mergeSegmentation {
  label 'cellpose'
  label 'minCpu'
  label 'extraMem'
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      tuple val(baseName), val(splittedFilenames), val(startHeight), path(images), path(inputImg), path('markers.csv')

  output:
    tuple val(baseName), path(inputImg), path('markers.csv'), path('*.ti{f,ff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_segmentation.py --in $images --out ${splittedFilenames}_masks.tiff --original $inputImg $args
    """
}