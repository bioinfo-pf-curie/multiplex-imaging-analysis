process pyramidize {
  label 'img_utils'
  label 'lowCpu'
  label "extraMem"
  label "highTime"

  input:
     tuple val(tag), val(meta), path(image)

  output:
    path("*.ome.tif")

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    pyramidize.py --in $image
    """
}