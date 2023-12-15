process mergeSegmentation {
  label 'cellpose'
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
      tuple val(base_name), val(splitted_filenames), val(startHeight), path(images), path(input_img), path('markers.csv')

  output:
    tuple val(base_name), path(input_img), path('markers.csv'), path('*.ti{f,ff}')

  when:
  task.ext.when == null || task.ext.when

  script:
    def out_name = splitted_filenames + "_masks.tiff"
    """
    merge_segmentation.py --in $images --out $out_name --original $input_img
    """
}