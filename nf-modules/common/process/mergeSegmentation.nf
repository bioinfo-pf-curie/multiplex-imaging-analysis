process mergeSegmentation {
  label 'cellpose'
  label 'minCpu'
  label 'maxMem'
  label 'higherTime'
  // maxMem is used untill I figure out if I can lower the memory from this step
  //container "${params.contPfx}${module.container}:${module.version}"


  input:
      tuple val(meta), path(images)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_segmentation.py --in $images --out ${meta.splittedName}_masks.tiff --original $meta.imagePath $args
    """
}