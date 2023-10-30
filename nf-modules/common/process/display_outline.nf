process displayOutline {
  label 'displayoutline'
  
  input:
    tuple val(filename), val(splitted_name), path(merge_tiff), val(tag), path(png_files)

  output:
    tuple val(filename), val(splitted_name), val(tag), path("*.tiff"), emit: out
    
  when:
  task.ext.when == null || task.ext.when

  script:
    """
    make_outlines.py --merge_tiff $merge_tiff --png_outline $png_files
    """
}
