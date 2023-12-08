process segmentation {
  label 'cellpose'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      tuple val(original_filename), val(splitted_filename), path(image), path(original_path)

  output:
    tuple val(original_filename), val(splitted_filename), path('*.npy')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    cellpose --channel_axis 0 --savedir . --verbose --chan 2 --chan2 1 --pretrained_model tissuenet --image_path $image
    """
}