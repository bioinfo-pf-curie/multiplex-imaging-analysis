import groovy.json.JsonOutput

process dataQC {
  label 'img_utils'
  label 'medCpu' 
  
  input:
  val(csvs)
  val(imgs)
  val(params)

  output:
  path 'figures/', emit: figures

  when:
  task.ext.when == null || task.ext.when

  script:
  def args = task.ext.args ?: ''
  // def jsonQC = JsonOutput.toJson(params.qualityControl)
  """
  mkdir -p figures/
  export NXF_ASSETS=${projectDir}/assets/
  quick_reporting.py --csv_path $csvs --img_path $imgs.imagePath --out_dir figures/ $args
  cp $csvs figures/
  """
}
