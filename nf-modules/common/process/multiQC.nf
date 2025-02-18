/*
 * MultiQC report
 */

process multiQC {
  label 'img_utils'
  label 'minCpu'
  label 'medMem'

  input:
  path ('figures/*')
  path (multiqcConfig)

  output:
  path "*_report.html", emit: report
  path "*_data", emit: data

  script:

  """
  multiqc figures/ -o \${PWD} -c ${multiqcConfig}
  """    
}

