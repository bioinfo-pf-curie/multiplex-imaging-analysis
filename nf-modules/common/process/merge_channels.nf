process mergeChannels {
  label 'mergechannels'
  label 'img_utils'
  
  input:
      tuple val(original_name), path(img), path(ch)

  output:
    tuple val(original_name), path("*.tif")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    mergechannels.py --in $img --channels $ch
    """
}