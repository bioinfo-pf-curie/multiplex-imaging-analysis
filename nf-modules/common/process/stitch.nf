process stitch {
  label 'img_utils'
  label 'minCpu'
  label 'highMem'
  
  input:
      tuple val(meta), path(images)

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def script = params.segmentation.name == 'cellpose' ? "stitch_flows.py" : "stitch_mesmer_output.py"
    """
    $script --in $images --out ${meta.originalName}.npy --original $meta.imagePath $args
    """
}