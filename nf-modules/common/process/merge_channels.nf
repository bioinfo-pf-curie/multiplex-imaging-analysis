process mergeChannels {
  label 'img_utils'
  
  input:
      tuple val(original_name), path(img), path(ch)
      each mode

  output:
    tuple val(original_name), path("*.tif")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    mergechannels.py --in $img --channels $ch --out ${original_name}_${mode}_merged.tif --norm $mode
    """
}