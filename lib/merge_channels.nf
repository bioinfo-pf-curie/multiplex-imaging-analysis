process mergechannels {
  input:
      tuple val(filename), path(img), path(ch)

  output:
    path("*.tif"), emit: out


  when:
    task.ext.when == null || task.ext.when

  script:
    """
    python mergechannels.py --in $img --channels $ch
    """
}