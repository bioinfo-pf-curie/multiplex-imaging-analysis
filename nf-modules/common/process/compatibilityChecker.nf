process compatibilityChecker {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  
  input:
    tuple val(meta), path(img), path(ch)

  output:
    tuple val(meta), path("*.tiff"), path(ch)

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    compatibility_checker.py --image $img --out ${meta.originalName}_checked.ome.tiff
    """
}