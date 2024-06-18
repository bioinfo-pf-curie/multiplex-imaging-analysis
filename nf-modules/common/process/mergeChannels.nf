process mergeChannels {
  label 'img_utils'
  label 'minCpu'
  label 'extraMem'
  label 'infiniteTime'
  
  input:
    tuple val(meta), path(img), path(ch)

  output:
    tuple val(meta), path("*.tiff")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_channels.py --in $img --channels $ch --out ${meta.originalName}.merged.tiff --segmentation_norm $args
    """
}