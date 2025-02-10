process compatibilityChecker {
  label 'img_utils'
  label 'lowCpu'
  label 'highMem'
  
  input:
    path(img)

  output:
    path("*.tiff")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    compatibility_checker.py --image $img --out "${img.getBaseName()}_checked.ome.tiff"
    """
}
