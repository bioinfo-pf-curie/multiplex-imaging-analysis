import argparse
import scimap as sm
import anndata as ad

def prepare_data(csv_path):
    return sm.pp.mcmicro_to_scimap(csv_path)

def unsupervised_clustering(adata, method='leiden'):
    return sm.tl.cluster(adata, method=method, use_raw=False, log=False)

def spatial_colormap(adata, color_by=['leiden_phenotype']):
    return sm.pl.spatial_scatterPlot (adata, colorBy=color_by, figsize=(3,3), s=0.7, fontsize=5, catCmap='Set1')

def save(adata, out_path):
    adata.write(out_path)

def write_report(imgs, report_path):
    with open(report_path, 'w') as out:
        for title, img in imgs: # or something like that
            out.write(title)
            out.write(img)

def main(csv_path, report_name, method):
    adata = prepare_data(csv_path)
    adata = unsupervised_clustering(adata, method=method)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--report_name', type=str, required=False, help="Output filepath")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()

    main(csv_path=args.csv_path, report_name=args.report_name, method=args.cluster_method)
