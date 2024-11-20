#!/usr/bin/env python

import argparse
# import scimap as sm
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import tifffile

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Table, Paragraph
import plotly.express as px
import plotly.graph_objects as go


class ScimapGraph:
    def __init__(self, csv_path, method):
        self.df = self.prepare_data(csv_path)
        self.compute_stuff(method)

        self.spatial_colormap(self.df, color_by=method, filepath="scimap/test_colormap.jpg")
        self.spatial_interaction(self.df, filepath="scimap/test_interaction.jpg")

        sm.pl.spatial_interaction(self.df, 
                          spatial_interaction='spatial_interaction_radius',
                          linewidths=0.75, linecolor='black', figsize=(5,4), saveDir="scimap", fileName="spatial_interaction.jpg")
        sm.pl.heatmap(self.df, groupBy=method, standardScale="column",
                    saveDir="scimap", fileName="cluster_phenotype.jpg")

        sm.pl.voronoi(self.df, color_by='spatial_aggregate_radius', 
                    voronoi_edge_color = 'black',
                    voronoi_line_width = 0.3, 
                    voronoi_alpha = 0.8, 
                    size_max=3000,
                    overlay_points=None,
                    saveDir="scimap", fileName="voronoi.jpg",
                    legend_size=6)
    
    @staticmethod
    def prepare_data(csv_path):
        return sm.pp.mcmicro_to_scimap(csv_path)
    
    def compute_stuff(self, method):
        self.unsupervised_clustering(method)
        self.df = sm.tl.spatial_interaction(self.df, phenotype=method,
                                    method='radius', 
                                    radius=70, 
                                    label='spatial_interaction_radius')
        self.df = sm.tl.spatial_aggregate(self.df, phenotype=method, 
                                        method='radius', radius=50, purity=80, label='spatial_aggregate_radius')

    
    @staticmethod
    def spatial_interaction(adata, spatial_interaction='spatial_interaction_radius', filepath="."):
        filepath = Path(filepath)
        return sm.pl.spatialInteractionNetwork(adata, spatial_interaction=spatial_interaction, figsize=(6,4), saveDir=filepath.parent, fileName=filepath.name)
    
    def unsupervised_clustering(self, method='leiden'):
        self.df = sm.tl.cluster(self.df, method=method, use_raw=False, log=False)

    @staticmethod
    def spatial_colormap(adata, color_by=['leiden_phenotype'], filepath=".", **kwargs):
        filepath = Path(filepath)
        return sm.pl.spatial_scatterPlot (adata, colorBy=color_by, s=3, 
                                        fontsize=5, catCmap='Set1', saveDir=filepath.parent, 
                                        fileName=filepath.name, **kwargs)

class GetBasicInfo:
    cn = {
        "area": "Area",
        "x": "X_centroid",
        "y": "Y_centroid",
        "extent": "Extent",
        "solidity": "Solidity",
        "eccentricity": "Eccentricity",
        "minoraxis": "MinorAxisLength",
        "majoraxis": "MajorAxisLength",
        "orientation": 'Orientation',
        'id': 'CellID'
    }

    def __init__(self, img_path, csv_path):
        self.df = pd.read_csv(csv_path)
        self.nb_cell = len(self.df)

        self.marker_cols = [col for col in self.df if col not in self.cn.values()]

        size_dis = self.make_size_distribution()
        marker_dis = self.make_markers_distribution()
        coexpr = self.make_co_expression()

        self.tiff = tifffile.TiffFile(img_path)
        self.get_fraction_segmented()

    def get_fraction_segmented(self):
        try:
            print(len(se))
            t = self.tiff.series[-1].asarray()
            flatten_thumbnail = np.apply_along_axis(np.mean, 0, t)
            print(flatten_thumbnail.shape)
        except:
            raise 

        # get original size compare to thumbnail
        thumbnail_factor = 2 ** (len(self.tiff.series) - 1)

        # separate tissue from background
        mask = np.where(thumbnail > 100, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.float32)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(thumbnail,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

        # calculate real area size of tissue
        tissue_area = mask.sum() * thumbnail_factor

        # sum all cells area
        segmented_area = self.df[self.cn["area"]].sum()

        self.fraction_segmented = segmented_area / tissue_area
        return self.fraction_segmented
    
    def make_size_distribution(self, fig_name="size_distribution.png"):
        fig = px.histogram(self.df, x=self.cn["area"])
        fig.show()

    def make_markers_distribution(self, dir_name="markers_distribution", fig_name="{marker}.png"):
        fig = px.violin(self.df, x=self.marker_cols)
        fig.update_yaxes(title_text="Markers")
        fig.update_xaxes(title_text="Intensities")
        fig.show()

    def make_co_expression(self, fig_name="scatter_matrix.png"):
        fig = px.scatter_matrix(self.df, dimensions=self.marker_cols)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.show()

        

class PDFReport:
    # header_style = ParagraphStyle('Hed0', fontSize=12, borderWidth=3, textColor="gray")
    # sub_header_style = ParagraphStyle('Hed3', fontSize=10, textColor="gray")

    def __init__(self, reportpath):
        self.path = reportpath

    def write_report(self):

        doc = SimpleDocTemplate(self.report_path, pagesize=letter)
        parts = [
                Paragraph(title, header_style),
                Paragraph("Info", sub_header_style),
                Table(info, colWidths=270, rowHeights=79)
        ]
        # parts.append(Image("scimap/spatial_interaction.jpg", width=400, height=560))
        # parts.append(Image("scimap/test_colormap.jpg", width=400, height=560))
        # parts.append(Image("scimap/test_interaction.jpg", width=400, height=560))
        # parts.append(Image("scimap/voronoi.jpg", width=400, height=560))
        doc.build(parts)
        

def main(csv_path, image_path, report_name, method):

    GetBasicInfo(image_path, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--img_path', type=str, required=True, help="path for original img")
    parser.add_argument('--report_name', type=str, required=True, help="Output filepath")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()

    main(csv_path=args.csv_path, image_path=args.img_path, report_name=args.report_name, method=args.cluster_method)
