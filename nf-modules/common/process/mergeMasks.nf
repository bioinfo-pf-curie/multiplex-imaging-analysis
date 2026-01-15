process mergeMasks {
  label "img_utils"
  label 'medCpu'

  // memory {MemoryUnit.of(Math.max(Math.min(meta.flowSize * 2, params.maxMemory.size), params.minMemory.size).toLong())}
  // memory {NFTools.computeRoundedMemoryGb((Float)(meta.flowSize * 2), task.attempt, params.minMemory, params.maxMemory)}
  memory {NFTools.computeRoundedMemoryGb(meta.flowSize * 2 as Float, task.attempt, params.minMemory, params.maxMemory)}

  input:
      tuple val(meta), path(partialMask, stageAs: "?/*"), val(diameters)

  output:
    tuple val(meta), path('*_masks.tiff')

  when:
  task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def diam_opts = ""
    def safe_diameters = (diameters instanceof Collection) ? diameters.findAll { it != null }.collect { it as Float } : []
    if (safe_diameters.size() > 0) {
        mean_diam = safe_diameters.sum() / safe_diameters.size()
        diam_opts += "--diameter ${mean_diam}"
    }
    
    """
    merge_masks.py --list_of_mask $partialMask --out "${meta.originalName}_masks.tiff" $diam_opts --original "${meta.imagePath}" $args
    """
}