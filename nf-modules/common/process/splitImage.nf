process splitImage {
  label 'img_utils'
  label 'minCpu'
  label 'lowMem'

  input:
    tuple val(meta), path(image)

  output:
    tuple stdout, val(meta), path('*.ti{f,ff}')

  when:
    task.ext.when == null || task.ext.when

  script:
    def maxHeight = params.segmentation.tileHeight ? 0 : task.memory.getBytes() / task.cpus
    def args = task.ext.args ?: ''
    """
    split_image.py --file_in $image --memory $maxHeight $args
    """
}