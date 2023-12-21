process splitImage {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
    tuple val(imgName), path(image)

  output:
    tuple val(imgName), path('*.ti{f,ff}'), path(image)

  when:
    task.ext.when == null || task.ext.when

  script:
    def maxHeight = task.memory.getBytes() / task.cpus
    def overlap = params.overlap ? " --overlap $params.overlap" : ""
    """
    split_image.py --file_in $image --memory $maxHeight $overlap
    """
}