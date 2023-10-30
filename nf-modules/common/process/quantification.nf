process quantification {
  label 'quantification'
  
  input:
      tuple val(filename), path(img), path(ch), val("mask"), path(mask)

  output:
    path("*.csv"), emit: out


  when:
    task.ext.when == null || task.ext.when

  script:
    """
    SingleCellDataExtraction.py --image $img --masks $mask --output . --channel_names $ch
    """
}