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
    def outName = quantif.name - ~/_masks\.csv/
    """
    quality_control.py --csv_path $quantif --out_path ${outName}_filtered_data.csv $args
    """
}
