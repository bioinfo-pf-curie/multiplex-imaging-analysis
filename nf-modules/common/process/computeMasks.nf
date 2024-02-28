process computeMasks {
  label 'cellpose'
  label 'minCpu'
  label 'higherTime'

  memory { MemoryUnit.of(Math.max(Math.min((meta.imgSize * 0.6).toFloat(), params.memoryMax), params.memoryMin).toLong()) }

  // maxMem is used untill I figure out if I can lower the memory from this step
  //container "${params.contPfx}${module.container}:${module.version}"


  input:
      tuple val(meta), path(flow)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    compute_masks.py --in $flow --out ${meta.originalName}_masks.tiff --original $meta.imagePath $args
    """
}