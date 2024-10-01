process makeReport {
  label "img_utils"
  label 'lowCpu'
  label 'lowMem'

  input:
    path(quantif)

  output:
    path("*.pdf")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    quick_reporting.py --csv_path $quantif --report_name ${quantif - ~/\.csv/}_report.pdf $args
    """
}
