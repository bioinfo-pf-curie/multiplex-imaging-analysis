process mergeImage {
  label 'mergeImage'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      tuple val(base_name), val(tag), path(images)
      tuple val(img_name), path(input_img), path('markers.csv')

  output:
    tuple val(base_name), val(tag), path('*.ti{f,ff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    merge_image.py --in $images --out $base_name --original $input_img
    """
}