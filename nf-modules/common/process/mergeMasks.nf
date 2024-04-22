process mergeMasks {
  label "geometrize"
  label 'lowCpu'
  label 'medMem'

  input:
      tuple val(meta), path(partialMask, stageAs: "?/*")

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_masks.py --list_of_mask $partialMask --out ${meta.originalName}_masks.tiff $args
    """
}