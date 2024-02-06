process stitch {
  label 'cellpose'
  label 'minCpu'
  label 'maxMem'
  label 'higherTime'
  // maxMem is used untill I figure out if I can lower the memory from this step
  //container "${params.contPfx}${module.container}:${module.version}"


  input:
      tuple val(meta), path(images)

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def script = params.segmentation.name == 'cellpose' ? "stitch_flows.py" : "stitch_masks.py"
    """
    $script --in $images --out ${meta.originalName}.npy --original $meta.imagePath $args
    """
}