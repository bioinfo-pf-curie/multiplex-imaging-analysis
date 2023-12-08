process quantification {
  label 'quantification'
  
  input:
      tuple val(filename), path(img), path(ch), path(mask), path(merged_img)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    SingleCellDataExtraction.py --image $img --masks $mask --output . --channel_names $ch
    """
}