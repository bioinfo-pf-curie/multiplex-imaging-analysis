/*
 * MultiQC report
 */

process multiQC {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'

  input:
  path ('figures/*')
  path (multiqcConfig)

  output:
  path "*_report.html", emit: report
  path "*_data", emit: data

  script:

  """
  multiqc . -o \${PWD} -c ${multiqcConfig}
  """    
}

