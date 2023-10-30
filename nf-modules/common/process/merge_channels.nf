process mergeChannels {
  label 'mergechannels'
  
  input:
      tuple val(original_name), val(splitted_name), path(img), path(ch)

  output:
    tuple val(original_name), val(splitted_name), path("*.tif"), emit: out


  when:
    task.ext.when == null || task.ext.when

  script:
    """
    mergechannels.py --in $img --channels $ch
    """
}