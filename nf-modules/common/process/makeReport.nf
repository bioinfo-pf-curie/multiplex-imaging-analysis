process makeReport {
  label "img_utils"
  label 'lowCpu'
  label 'lowMem'

  input:
    tuple val(meta), path(quantif)

  output:
    path("*.pdf")

  when:
    task.ext.when == null || task.ext.when

  script:
    def args = task.ext.args ?: ''
    """
    quick_reporting.py --csv_path $quantif --img_path $meta.imagePath --report_name ${quantif - ~/\.csv/}_report.pdf $args
    """
}
