process computeMasks {
  label 'cellpose'
  label 'minCpu'
  label 'maxMem' // on pourrait remplacer ça par 2.5 * taille de l'image original
  label 'higherTime'
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
    compute_masks.py --in $flow --out ${meta.splittedName}_masks.tiff --original $meta.imagePath $args
    """
}