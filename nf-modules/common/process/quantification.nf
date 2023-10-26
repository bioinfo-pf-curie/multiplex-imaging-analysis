process quantification {
  label 'quantification'
  
  input:
      tuple val(filename), path(img), path(ch)
      path(markers)

  output:
    path("*.csv"), emit: out


  when:
    task.ext.when == null || task.ext.when

  script:
    """
    SingleCellDataExtraction.py --image $img --masks $markers --output . --channel_names $ch
    """
}