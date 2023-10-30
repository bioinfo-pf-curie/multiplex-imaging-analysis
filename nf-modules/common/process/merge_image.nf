process mergeImage {
  label 'mergeImage'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      tuple val(base_name), val(imgs_names), val(tag), path(images), path(input_img), path('markers.csv')

  output:
    tuple val(base_name), path(input_img), path('markers.csv'), val(tag), path('*.ti{f,ff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    def out_name = base_name + "_" + tag + ".tiff"
    """
    merge_image.py --in $images --out $out_name --original $input_img
    """
}