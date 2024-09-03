import nextflow.util.MemoryUnit as MemoryUnit

process splitImage {
  label 'img_utils'
  label 'lowCpu'
  label 'lowMem'

  input:
    tuple val(meta), path(image)

  output:
    tuple stdout, val(meta), path('*.ti{f,ff}')

  when:
    task.ext.when == null || task.ext.when

  script:
    // if params.segmentation.tileHeight is set, it will be passed into args
    // availableMem need to be scaled down if diameter if lower than 30 because of rescaling tile...
    def availableMem = params.segmentation.tileHeight ? 0 : (params.segmentation.memory instanceof MemoryUnit ? params.segmentation.memory : MemoryUnit.of(params.segmentation.memory)) 
    if (availableMem instanceof MemoryUnit) {
      availableMem = availableMem.getBytes()
    }
    def args = task.ext.args ?: ''
    def scaling = (params.segmentation.diameter / 30) ** 2 // default is 1
    """
    split_image.py --file_in $image --memory $availableMem --scaling $scaling $args
    """
}