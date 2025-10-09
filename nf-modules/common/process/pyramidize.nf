process pyramidize {
  label 'pyramidize'
  label 'lowCpu'
  label "infiniteTime"

  memory {
    NFTools.computeRoundedMemoryGb((Float)(image.size() * ((tag == "merged") ? 0.3 : 0.6)), task.attempt, params.minMemory, params.maxMemory)
  }

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