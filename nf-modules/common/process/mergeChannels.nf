process mergeChannels {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  
  input:
      tuple val(originalName), path(img), path(ch)
      each mode

  output:
    tuple val(originalName), path("*.tiff")

  when:
    task.ext.when == null || task.ext.when

  script:
    def modeOpt = mode != "" ? "--norm $mode" : "" 
    """
    merge_channels.py --in $img --channels $ch --out ${originalName}_${mode}.merged.tiff $modeOpt
    """
}