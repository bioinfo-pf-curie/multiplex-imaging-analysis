process displayOutline {
  label 'displayoutline'
  label 'img_utils'
  
  input:
    tuple val(filename), path(original_path), path('markers.csv'), path(mask), path(merge_tiff)

  output:
    tuple val(filename), path("*.ti{f,ff}")
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $merge_tiff --mask $mask
    """
}
