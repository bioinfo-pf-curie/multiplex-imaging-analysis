process quantification {
  label 'quantification'
  label "minCpu"
  label 'extraMem'
  label "highTime"

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