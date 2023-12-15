process quantification {
  label 'quantification'

  publishDir saveAs: "${filename}_data.csv"
  // can't use filename in config (or more likely idk how)
  
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