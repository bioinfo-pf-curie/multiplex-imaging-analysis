process splitImage {
  label 'splitImage'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
    tuple val(img_name), path(image)

  output:
    tuple val(img_name), path('*.ti{f,ff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    split_image.py $image 
    """
}