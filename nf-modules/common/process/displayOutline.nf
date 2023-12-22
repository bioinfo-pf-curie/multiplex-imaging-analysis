process displayOutline {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  
  input:
    tuple val(meta), path(mask)

  output:
    tuple val(meta), path("*_clear_outline.tiff")
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $meta.imagePath --mask $mask --all-channels --out ${meta.originalName}_clear_outline.tiff
    """
}
