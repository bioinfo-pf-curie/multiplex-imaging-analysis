process pyramidize {
  label 'pyramidize'
  label 'lowCpu'
  label "infiniteTime"

  memory {MemoryUnit.of(Math.max(Math.min((image.size() as Float) * 0.3 * task.attempt, params.maxMemory.size), params.minMemory.size * 2).toLong())}

  input:
     tuple val(tag), val(meta), path(image)

  output:
    tuple val(tag), val(meta), path("*.ome.tif")

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    pyramidize.py --in $image $args
    """
}