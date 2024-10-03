process ome2panel {
  label "img_utils"
  label 'lowCpu'
  label 'lowMem'

  input:
    path(img)

  output:
    path("*.csv")

  when:
    task.ext.when == null || task.ext.when

  script:
    """
    ome2panel.py --image $img --out ${img.getBaseName()}.csv --notSegmented $params.channelNotSegmented
    """
}