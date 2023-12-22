process segmentation {
  label 'cellpose'
  label 'lowCpu'
  label 'medMem'

  input:
    tuple val(meta), path(image)

  output:
    tuple val(meta), path('*.npy')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image
    """
}