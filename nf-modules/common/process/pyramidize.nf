process pyramidize {
  label 'pyramidize'
  label 'minCpu'
  label "highMem"

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