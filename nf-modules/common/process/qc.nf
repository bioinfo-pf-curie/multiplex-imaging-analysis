process qc {
  label "img_utils"
  label 'lowCpu'
  label 'lowMem'

  input:
    path(quantif)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    quality_control.py --csv_path $quantif --out_path ${quantif - ~/panel\.csv/}_filtered_panel.csv $args
    """
}
