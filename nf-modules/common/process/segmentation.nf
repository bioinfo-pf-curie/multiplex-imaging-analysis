process segmentation {
  label "${params.segmentation.name}"
  label 'lowCpu'
  label 'medMem'
  label 'onlyLinux'

  input:
    tuple val(meta), path(image)

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def cellpose = "cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image"
    def mesmer = "wrapper_mesmer.py --squeeze --output-directory . --output-name cell.tif"
    def script = params.segmentation.name == 'cellpose' ? cellpose : mesmer
    """
    $script
    """
}