process computeMasks {
  label "${params.segmentation.name == 'cellpose'? 'cellpose': 'img_utils'}"
  label 'highCpu'
  label 'infiniteTime'
  label 'onlyLinux' // only for geniac lint...

  memory {params.segmentation.name == 'cellpose'? MemoryUnit.of(Math.max(Math.min(meta.imgSize * 0.4, params.maxMemory.size), params.minMemory.size).toLong()) : params.maxMemory}
  // take 40% of the size of image input with a minimum of 2GB and a max of 190GB 

  input:
      tuple val(meta), path(flow)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def memo = Math.min(meta.imgSize * 0.4, params.maxMemory.size)
    def args = task.ext.args ?: ''
    def cellpose = "compute_masks.py"
    def mesmer = "compute_mesmer.py"
    def script = params.segmentation.name == 'cellpose' ? cellpose : mesmer
    """
    echo $memo
    $script --in $flow --out ${meta.originalName}_masks.tiff --original $meta.imagePath $args
    """
}