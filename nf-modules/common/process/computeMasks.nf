process computeMasks {
  label 'cellpose'
  label 'lowCpu'
  label 'infiniteTime'

  memory { MemoryUnit.of(Math.max(Math.min((meta.imgSize * 0.4).toFloat(), params.memoryMax), params.memoryMin).toLong()) }
  // take 40% of the size of image input with a minimum of 2GB and a max of 190GB 

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