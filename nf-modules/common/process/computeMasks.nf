process computeMasks {
  label "${params.segmentation.name == 'cellpose'? 'cellpose': 'img_utils'}"
  label 'medCpu'
  label 'infiniteTime'
  label 'onlyLinux' // only for geniac lint...

  memory {params.segmentation.name == 'cellpose'? MemoryUnit.of(Math.max(Math.min(meta.imgSize * 2, params.maxMemory.size), params.minMemory.size).toLong()) : Math.min(params.maxMemory.size, (128.GB).getBytes())}
  // take 40% of the size of image input with a minimum of 2GB and a max of 190GB 

  input:
      tuple val(meta), path(flow)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def script = params.segmentation.name == 'cellpose' ? "compute_masks.py" : "compute_mesmer.py"
    """
    $script --in $flow --out ${meta.originalName}_masks.tiff --mean_cell_diam $meta.diameter --original $meta.imagePath $args
    """
}