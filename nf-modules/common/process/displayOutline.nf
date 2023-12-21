process displayOutline {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  
  input:
    tuple val(filename), path(originalPath), path('markers.csv'), path(mask), path(mergeTiff)

  output:
    tuple val(filename), path("*.ti{f,ff}")
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $originalPath --mask $mask --all-channels --out ${mergeTiff.getSimpleName()}_clear_outline.tiff
    """
}
