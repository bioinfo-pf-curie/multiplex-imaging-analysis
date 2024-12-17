process qc {
  label "img_utils"
  label 'lowCpu'
  label 'lowMem'

  input:
    tuple val(meta), path(quantif)

  output:
    tuple val(meta), path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    def outName = quantif.baseName - ~/_data$/
    def roi = params.qualityControl.ROIPath ? "--region_of_interest_geojson_path $params.qualityControl.ROIPath": ""
    def excl = params.qualityControl.excludedPath ? "--excluded_region_geojson_path $params.qualityControl.excludedPath": ""
    """
    quality_control.py --csv_path $quantif --out_path ${outName}_filtered_data.csv $roi $excl $args
    """
}
