process stitchFlows {
  label 'cellpose'
  label 'minCpu'
  label 'highMem'
  
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