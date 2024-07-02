process mergeMasks {
  label "img_utils"
  label 'medCpu'
  memory {MemoryUnit.of(Math.max(Math.min(meta.imgSize * 0.6, params.maxMemory.size), (8.GB).size).toLong())}

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