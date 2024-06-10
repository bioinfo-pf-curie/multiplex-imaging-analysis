process mergeMasks {
  label "img_utils"
  label 'lowCpu'
  memory {MemoryUnit.of(Math.max(Math.min(meta.imgSize * 0.6, params.maxMemory.size), params.minMemory.size).toLong())}

  input:
      tuple val(meta), path(partialMask, stageAs: "?/*")

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    merge_masks.py --list_of_mask $partialMask --out ${meta.originalName}_masks.tiff --original $meta.imagePath $args
    """
}