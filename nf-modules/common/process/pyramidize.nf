process pyramidize {
  label 'pyramidize'
  label 'minCpu'
  label "medMem"

  input:
     tuple val(tag), val(filename), path(image)

  output:
    path("*.ome.tif")

  when:
  task.ext.when == null || task.ext.when

  script:
    """
    pyramidize.py --in $image
    """
}