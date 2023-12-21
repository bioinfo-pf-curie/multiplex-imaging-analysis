process mergeChannels {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  
  input:
      tuple val(originalName), path(img), path(ch)

  output:
    tuple val(originalName), path("*.tiff")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_channels.py --in $img --channels $ch --out ${originalName}.merged.tiff $args
    """
}