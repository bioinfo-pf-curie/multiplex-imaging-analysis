process mask2geojson {
  label 'img_utils'
  label 'medCpu'

  // memory {MemoryUnit.of(Math.max(Math.min(meta.imgSize * 1.2, params.maxMemory.size), params.minMemory.size).toLong())}
  memory {NFTools.computeRoundedMemoryGb(meta.imgSize * 1.2, task.attempt, params.minMemory, params.maxMemory)}

  input:
    tuple val(meta), path(image)

  output:
    tuple val(meta), path('*.geojson')

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    mask2geojson.py --mask $image --out "${meta.originalName}.geojson"
    """
}