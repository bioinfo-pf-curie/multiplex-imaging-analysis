process mergeMasks {
  label "img_utils"
  label 'medCpu'
  memory {MemoryUnit.of(Math.max(Math.min(meta.flowSize * 2, params.maxMemory.size), params.minMemory.size).toLong())}

  input:
      tuple val(meta), path(partialMask, stageAs: "?/*"), val(diameters)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def meanCellDiam = diameters.sum() / diameters.size()
    """
    merge_masks.py --list_of_mask $partialMask --out ${meta.originalName}_masks.tiff --diameter $meanCellDiam --original $meta.imagePath $args
    """
}