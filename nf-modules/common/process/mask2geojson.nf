process mask2geojson {
  label 'img_utils'
  label 'medCpu'
  label 'highMem'

  input:
    tuple val(meta), path(image)

  output:
    tuple val(meta), path('*.geojson')

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    mask2geojson.py --mask $image --out ${meta.originalName}.geojson
    """
}