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
                csvs: quant[1]
                imgs: quant[0]
        }
        dataQC(
            dataCh.csvs,
            dataCh.imgs,
            params
        )
        
        multiQC(
            dataQC.out.figures,
            multiqcConfigCh
        )

        qcreportCh = multiQC.out.report

    emit:
        reportCh = qcreportCh
}
