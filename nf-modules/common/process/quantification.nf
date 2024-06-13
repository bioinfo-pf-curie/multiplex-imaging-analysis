process quantification {
  label 'img_utils'
  label "minCpu"
  label "infiniteTime"

  memory {MemoryUnit.of(Math.max(Math.min(meta.imgSize * 0.3, params.maxMemory.size), params.minMemory.size).toLong())}

  input:
      tuple val(meta), path(mask)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    single_cell_data_extraction.py --image $meta.imagePath --masks $mask --output . --channel_names $meta.markersPath $args
    """
}