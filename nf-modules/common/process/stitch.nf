process stitch {
  label 'img_utils'
  label 'minCpu'
  label 'highMem'
  
  input:
      tuple val(meta), path(images)
      val segmenterConfig

  output:
    tuple val(meta), path('*.{npy,tiff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    $segmenterConfig.stitch --in $images --original $meta.imagePath $args
    """
}