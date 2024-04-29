process displayOutline {
  label 'img_utils'
  label 'medCpu'
  label 'extraMem'
  
  input:
    tuple val(meta), path(mask), path(merge)

  output:
    tuple val(meta), path("*_outline.tiff")
    
  when:
  task.ext.when == null || task.ext.when

  script:
    def inpt = params.outline == "merged" ? merge : meta.imagePath
    """
    make_outlines.py --merge-tiff $inpt --mask $mask --all-channels --out ${meta.originalName}_outline.tiff
    """
}
