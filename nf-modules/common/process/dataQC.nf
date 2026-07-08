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
  def convertedMap = params.qualityControl.collectEntries { key, value ->
          // Vérifier si la valeur est compatible JSON, sinon la convertir en str
          if (value instanceof String || value instanceof Boolean || value instanceof Number || value == null) {
              [key, value]  // Valeur compatible avec JSON
          } else {
              [key, value.toString()]  // Valeur non compatible avec JSON, convertie en str
          }
      }
  def jsonQC = JsonOutput.toJson(convertedMap)
  """
  mkdir -p figures/
  export NXF_ASSETS=${projectDir}/assets/
  quick_reporting.py --csv_path "$csvs" --img_path "$imgs" --parms '$jsonQC' --out_dir figures/ $args
  cp "$csvs" figures/
  """
}
