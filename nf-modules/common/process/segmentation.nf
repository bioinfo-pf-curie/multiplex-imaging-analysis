process segmentation {
  label 'cellpose'
  label 'lowCpu'
  label 'medMem'

  input:
      tuple val(originalFilename), val(splittedFilename), val(startHeight), path(image), path(originalPath)

  output:
    tuple val(originalFilename), val(splittedFilename), val(startHeight), path('*.npy')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image
    """
}