process stitchFlows {
  label 'cellpose'
  label 'minCpu'
  label 'maxMem'
  label 'higherTime'
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
    stitch_flows.py --in $images --out ${meta.splittedName}.npy --original $meta.imagePath $args
    """
}