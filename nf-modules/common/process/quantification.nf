process quantification {
  label 'img_utils'
  label "minCpu"
  label "infiniteTime"

  // memory {MemoryUnit.of(Math.max(Math.min((mask.size() as Float) * 5, params.maxMemory.size), params.minMemory.size).toLong())}
  memory {NFTools.computeRoundedMemoryGb((mask.size() as Float) * 5, task.attempt, params.minMemory, params.maxMemory)}

  input:
      tuple val(meta), path(mask)

  output:
    tuple val(meta), path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    single_cell_data_extraction.py --image "$meta.imagePath" --masks $mask --output . --channel_names "$meta.markersPath" $args
    """
}