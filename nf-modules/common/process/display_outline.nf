process displayOutline {
  label 'displayoutline'
  
  input:
      tuple val(filename), path(merge_tiff)
      tuple val(filename), val(tag), path(png_files)

  output:
    tuple val(filename), val(tag), path("*.tiff"), emit: out
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $merge_tiff --png_outline $png_files
    """
}
