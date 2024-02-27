process splitImage {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'
  //conda "${projectDir}/env/conda_env.yml"
  //container "${params.contPfx}${module.container}:${module.version}"

  input:
    tuple val(meta), path(image)

  output:
    tuple stdout, val(meta), path('*.ti{f,ff}')

  when:
    task.ext.when == null || task.ext.when

  script:
    def maxHeight = params.titleHeight ? 0 : task.memory.getBytes() / task.cpus
    def args = task.ext.args ?: ''
    """
    split_image.py --file_in $image --memory $maxHeight $args
    """
}