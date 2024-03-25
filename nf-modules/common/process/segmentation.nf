process segmentation {
  label "${params.segmentation.name}"
  label 'lowCpu'
  label 'medMem'
  label 'onlyLinux' // only for geniac lint...

  input:
    tuple val(meta), path(image)

  output:
    tuple val(meta), path(params.segmentation.name == 'cellpose'? '*.npy': '*.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def cellpose = "cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image"
    def mesmer = "wrapper_mesmer.py --squeeze --output-directory . --output-name ${meta.splittedName}_masks.tiff --nuclear-image $image --membrane-image $image --membrane-channel 1"
    def script = params.segmentation.name == 'cellpose' ? cellpose : mesmer
    """
    $script $task.ext.args
    """
}