process stitchFlows {
  label 'cellpose'
  label 'minCpu'
  label 'higherTime'

  memory { MemoryUnit.of(Math.max(Math.min((meta.imgSize * 0.3).toFloat(), params.memoryMax), params.memoryMin).toLong()) }
  // take 30% of the size of image input with a minimum of 2GB and a max of 190GB 
  //container "${params.contPfx}${module.container}:${module.version}"


  input:
      tuple val(meta), path(images)

  output:
    tuple val(meta), path('*.npy')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    stitch_flows.py --in $images --out ${meta.originalName}.npy --original $meta.imagePath $args
    """
}