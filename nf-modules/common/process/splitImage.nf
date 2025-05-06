import nextflow.util.MemoryUnit as MemoryUnit

process splitImage {
  label 'img_utils'
  label 'lowCpu'
  label 'medMem'

  input:
    tuple val(meta), path(image)
    val segmenterConfig

  output:
    tuple stdout, val(meta), path('*.ti{f,ff}')

  when:
    task.ext.when == null || task.ext.when

  script:
    // if params.segmentation.tileHeight is set, it will be passed into args
    // availableMem need to be scaled down if diameter if lower than 30 because of rescaling tile...
    def scaling = (segmenterConfig.diameter / 30) ** 2 // default is 1
    def args = "--overlap $segmenterConfig.overlap --scaling $scaling --memory "
    if (segmenterConfig.tileHeight) {
      args += "0 --height $segmenterConfig.tileHeight "
    } else {
      def availableMem = (segmenterConfig.memory instanceof MemoryUnit ? segmenterConfig.memory : MemoryUnit.of(segmenterConfig.memory)).getBytes()
      args += "$availableMem "
    }
    args += task.ext.args ?: ''
    """
    split_image.py --file_in $image $args
    """
}