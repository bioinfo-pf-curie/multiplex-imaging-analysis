process mergeChannels {
  label 'img_utils'
  
  input:
      tuple val(original_name), path(img), path(ch)
      each mode

  output:
    tuple val(original_name), path("*.tiff")

  when:
    task.ext.when == null || task.ext.when

  script:
    def modeOpt = mode != "" ? "--norm $mode" : "" 
    """
    merge_channels.py --in $img --channels $ch --out ${original_name}_${mode}.merged.tiff $modeOpt
    """
}