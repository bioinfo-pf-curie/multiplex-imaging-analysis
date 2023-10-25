process displayoutline {
  input:
      path(merge_tiff)
      path(png_files)

  output:
    path("*.tiff"), emit: out

  script:
    """
    python make_outlines.py --merge_tiff $merge_tiff --png_outline $png_files
    """
}
