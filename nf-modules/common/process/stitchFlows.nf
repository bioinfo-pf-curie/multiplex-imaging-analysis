process stitchFlows {
  label 'cellpose'
  label 'minCpu'
  label 'higherTime'

  memory { MemoryUnit.of(Math.max(Math.min((int)(meta.imgSize * 0.3), 190000000000), 2000000000)) }
  // take 30% of the size of image input with a minimum of 2GB and a max of 190GB 
  // maxMem is used untill I figure out if I can lower the memory from this step
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