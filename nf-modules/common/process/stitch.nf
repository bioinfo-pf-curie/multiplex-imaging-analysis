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
    def outname = params.segmentation.name == 'cellpose' ? "${meta.originalName}.npy" : "${meta.originalName}.tiff"
    """
    $script --in $images --out $outname --original $meta.imagePath $args
    """
}