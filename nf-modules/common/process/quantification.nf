process quantification {
  label 'quantification'
  label "minCpu"
  label 'highMem'

  publishDir saveAs: "${filename}_data.csv"
  // can't use filename in config (or more likely idk how)
  
  input:
      tuple val(meta), path(mask)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    single_cell_data_extraction.py --image $meta.imagePath --masks $mask --output . --channel_names $meta.markersPath
    """
}