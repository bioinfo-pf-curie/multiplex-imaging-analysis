process pyramidize {
  label 'pyramidize'
  label 'minCpu'
  label "highTime"

  memory {MemoryUnit.of(Math.max(Math.min(meta.imgSize * 0.2, params.maxMemory.size), params.minMemory.size).toLong())}

  input:
     tuple val(tag), val(meta), path(image)

  output:
    path("*.ome.tif")

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    pyramidize.py --in $image
    """
}