process spatialData {
  label 'img_utils'
  label 'medCpu'
  label 'medMem'

  input:
    tuple val(meta), path(mask), path(quantification)

  output:
    tuple val(meta), path('*.zarr')

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    ome2spatial_data.py --image ${meta.imagePath} --mask $mask --quantification $quantification --panel ${meta.markersPath} --out "${meta.originalName}.zarr"
    """
}