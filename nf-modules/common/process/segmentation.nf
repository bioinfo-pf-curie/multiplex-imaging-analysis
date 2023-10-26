process segmentation {
  label 'cellpose'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      path(image)

  output:
    path('*.ti{f,ff}')
    path('*.png')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    cellpose --channel_axis 0 --save_tif --savedir . --verbose \
    --no_npy --chan 2 --chan2 1 --pretrained_model tissuenet --save_outlines --image_path $image
    """
}