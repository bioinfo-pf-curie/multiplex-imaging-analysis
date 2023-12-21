process quantification {
  label 'quantification'
  label "minCpu"
  label 'highMem'

  publishDir saveAs: "${filename}_data.csv"
  // can't use filename in config (or more likely idk how)
  
  input:
      tuple val(filename), path(img), path(ch), path(mask), path(mergedImg)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    single_cell_data_extraction.py --image $img --masks $mask --output . --channel_names $ch
    """
}