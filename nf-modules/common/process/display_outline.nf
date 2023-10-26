process displayoutline {
  label 'displayoutline'
  
  input:
      path(merge_tiff)
      path(png_files)

  output:
    path("*.tiff"), emit: out
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $merge_tiff --png_outline $png_files
    """
}
