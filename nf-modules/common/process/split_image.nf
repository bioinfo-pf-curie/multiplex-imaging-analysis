process splitImage {
  label 'splitImage'
  label 'img_utils'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
    tuple val(img_name), path(image)

  output:
    tuple val(img_name), path('*.ti{f,ff}'), path(image)

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    split_image.py $image 
    """
}