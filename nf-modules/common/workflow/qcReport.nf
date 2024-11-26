/* 
 * QC report for MIA
 */
include { dataQC } from '../process/dataQC' 
include { multiQC } from '../process/multiQC'


workflow qcFlow {

    take:
        quantificationData
        params

    main:

        multiqcConfigCh = Channel.fromPath(params.multiqcConfig)
        dataCh = quantificationData.multiMap{
            quant -> 
                csvs: quant.value[0][1]
                imgs: quant.value[0][0]
        }
        dataQC(
            dataCh.csvs,
            dataCh.imgs,
            params
        )
        
        // figCh = Channel.empty()
        multiQC(
            dataQC.out.figures,
            multiqcConfigCh
        )

        qcreportCh = multiQC.out.report

    emit:
        reportCh = qcreportCh
}
