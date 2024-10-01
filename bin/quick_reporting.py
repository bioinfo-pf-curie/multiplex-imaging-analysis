#!/usr/bin/env python

import argparse
import scimap as sm
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image

def prepare_data(csv_path):
    return sm.pp.mcmicro_to_scimap(csv_path)

def unsupervised_clustering(adata, method='leiden'):
    return sm.tl.cluster(adata, method=method, use_raw=False, log=False)

def compute_stuff(adata, method):
    adata = unsupervised_clustering(adata, method)
    adata = sm.tl.spatial_interaction(adata, phenotype=method,
                                  method='radius', 
                                  radius=70, 
                                  label='spatial_interaction_radius')
    adata = sm.tl.spatial_aggregate(adata, phenotype=method, 
                                    method='radius', radius=50, purity=80, label='spatial_aggregate_radius')
    return adata

def spatial_colormap(adata, color_by=['leiden_phenotype'], filepath=".", **kwargs):
    filepath = Path(filepath)
    return sm.pl.spatial_scatterPlot (adata, colorBy=color_by, s=3, 
                                      fontsize=5, catCmap='Set1', saveDir=filepath.parent, 
                                      fileName=filepath.name, **kwargs)

def spatial_interaction(adata, spatial_interaction='spatial_interaction_radius', filepath="."):
    filepath = Path(filepath)
    return sm.pl.spatialInteractionNetwork(adata, spatial_interaction=spatial_interaction, figsize=(6,4), saveDir=filepath.parent, fileName=filepath.name)

def save(adata, out_path):
    adata.write(out_path)

def write_report(report_path):

    doc = SimpleDocTemplate(report_path, pagesize=letter)
    parts = []
    parts.append(Image("scimap/spatial_interaction.jpg", width=400, height=560))
    parts.append(Image("scimap/test_colormap.jpg", width=400, height=560))
    parts.append(Image("scimap/test_interaction.jpg", width=400, height=560))
    parts.append(Image("scimap/voronoi.jpg", width=400, height=560))
    doc.build(parts)

def main(csv_path, report_name, method):
    adata = prepare_data(csv_path)
    adata = compute_stuff(adata, method=method)
    spatial_colormap(adata, color_by=method, filepath="scimap/test_colormap.jpg")
    spatial_interaction(adata, filepath="scimap/test_interaction.jpg")
    sm.pl.spatial_interaction(adata, 
                          spatial_interaction='spatial_interaction_radius',
                          linewidths=0.75, linecolor='black', figsize=(5,4), saveDir="scimap", fileName="spatial_interaction.jpg")
    sm.pl.heatmap(adata, groupBy=method, standardScale="column",
                  saveDir="scimap", fileName="cluster_phenotype.jpg")

    sm.pl.voronoi(adata, color_by='spatial_aggregate_radius', 
                 voronoi_edge_color = 'black',
                 voronoi_line_width = 0.3, 
                 voronoi_alpha = 0.8, 
                 size_max=3000,
                 overlay_points=None,
                 saveDir="scimap", fileName="voronoi.jpg",
                 legend_size=6)

    write_report(report_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--report_name', type=str, required=True, help="Output filepath")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()

    main(csv_path=args.csv_path, report_name=args.report_name, method=args.cluster_method)
