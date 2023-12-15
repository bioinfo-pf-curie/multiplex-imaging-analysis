process segmentation {
  label 'cellpose'

  input:
      tuple val(original_filename), val(splitted_filename), val(startHeight), path(image), path(original_path)

  output:
    tuple val(original_filename), val(splitted_filename), val(startHeight), path('*.npy')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image
    """
}