#!/usr/bin/env python

import argparse
# import scimap as sm
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import tifffile
import plotly.express as px
import plotly.io as pio

from utils import min_max_norm

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Image, Table, Paragraph, Spacer, PageBreak


pio.templates.default = "plotly_white"


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

        self.size_dis = self.make_size_distribution(height=650)
        self.marker_dis = self.make_markers_distribution(height=650)
        coexpr_size = max(600, 70 * len(self.marker_cols))
        self.coexpr = self.make_co_expression(height=coexpr_size, width=coexpr_size)

        self.tiff = tifffile.TiffFile(img_path)
        if self.tiff.series[0].is_pyramidal:
            self.thumbnail = self.tiff.series[0].levels[-1].asarray()
            i,a = np.quantile(self.thumbnail, [0.01,0.99])
            self.thumbnail = min_max_norm(self.thumbnail, i, a, output_max=255)
        else: 
            self.thumbnail = None

        self.segmented_fraction = self.get_fraction_segmented()
        self.th_img = tiff2rgb(self.thumbnail)
        tifffile.imwrite('test_thumb.tiff', self.thumbnail)

    def get_fraction_segmented(self):
        if self.thumbnail is None:
            raise ValueError('No thumbnail to compute')
        try:
            flatten_thumbnail = self.thumbnail.mean(axis=0).astype('uint8')
        except:
            raise 

        # get original size compare to thumbnail
        thumbnail_factor = 2 ** ((len(self.tiff.series[0].levels) - 1) * 2) # *2 for area
        # separate tissue from background
        mask = np.where(flatten_thumbnail > np.quantile(flatten_thumbnail,0.2), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        print(np.unique(mask, return_counts=True))
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(cv2.cvtColor(flatten_thumbnail, cv2.COLOR_GRAY2RGB),mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

        # calculate real area size of tissue
        tissue_area = np.where(mask == cv2.GC_PR_FGD, 1, 0).sum() * thumbnail_factor 
        self.tissue_fraction = tissue_area / (np.multiply(*mask.shape) * thumbnail_factor)
        # sum all cells area
        segmented_area = self.df[self.cn["area"]].sum()

        return segmented_area / tissue_area
    
    def make_size_distribution(self, fig_name="size_distribution.png", *args, **kwargs):
        fig = px.histogram(self.df, x=self.cn["area"])
        fig.write_image(fig_name, *args, **kwargs)
        return fig_name

    def make_markers_distribution(self, fig_name="markers_distribution.png", *args, **kwargs):
        fig = px.violin(self.df, x=self.marker_cols)
        fig.update_yaxes(title_text="Markers")
        fig.update_xaxes(title_text="Intensities")
        fig.write_image(fig_name, *args, **kwargs)
        return fig_name

    def make_co_expression(self, fig_name="scatter_matrix.png", *args, **kwargs):
        fig = px.scatter_matrix(self.df, dimensions=self.marker_cols)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(font_size=6)
        fig.write_image(fig_name, *args, **kwargs)
        return fig_name


class PDFReport:

    def __init__(self, reportpath, pagesize=A4):
        self.styles = getSampleStyleSheet()
        self.path = reportpath
        self.doc = SimpleDocTemplate(self.path, pagesize=pagesize)
        self.doc_width, self.doc_height = pagesize
        self.parts = []

    def p(self, text, style='BodyText'):
        self.parts.append(Paragraph(text, style=self.styles[style]))

    def header(self, title):
        self.p(title, 'h1')

    def spacer(self, height=100, width=None):
        if width is None:
            width = self.doc_width
        self.parts.append(Spacer(width=width, height=height))

    def img(self, src, width=None, height=None):
        img = Image(src, width=width, height=height)
        img.vAlign = "MIDDLE"
        self.parts.append(img)

    def page_break(self):
        self.parts.append(PageBreak())

    def write_report(self):
        # parts.append(Image("scimap/spatial_interaction.jpg", width=400, height=560))
        # parts.append(Image("scimap/test_colormap.jpg", width=400, height=560))
        # parts.append(Image("scimap/test_interaction.jpg", width=400, height=560))
        # parts.append(Image("scimap/voronoi.jpg", width=400, height=560))
        self.doc.build(self.parts)

def tiff2rgb(img, out_path="thumbnail.png"):
    color_cycle = [[int(h.strip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)] 
                   for h in ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']]
    result = np.transpose(np.stack([img[0], img[1], img[2]]), (1,2,0)) # take the first three channels as RGB

    # and merge the rest
    for channel in range(3, img.shape[0]):
        tmp_img = np.transpose(np.stack([img[channel]] * 3) * np.array(color_cycle[channel % len(color_cycle)])[:,None,None], (1,2,0))
        alpha = 1 / (channel + 1)
        result = cv2.addWeighted(result, 1-alpha, tmp_img, alpha, 0)
    cv2.imwrite(out_path, result.astype('uint8'))
    return out_path


def main(csv_path, image_path, report_name, method):
    
    info = GetBasicInfo(image_path, csv_path)

    mypdf = PDFReport(report_name)
    
    mypdf.header(Path(csv_path).stem)
    mypdf.img(info.th_img)#, width=200, height=200)
    mypdf.spacer()
    mypdf.p('Info', 'h3')
    mypdf.p(f"""
- {info.nb_cell} cell{'s' if info.nb_cell > 1 else ''} found<br />
- Tissue / Background area : {info.tissue_fraction*100:.02f} %<br />
- Segmented Fraction : {info.segmented_fraction*100:.02f} %<br />
""")
    mypdf.page_break()
    mypdf.p('Size Distribution', 'h3')
    mypdf.img(info.size_dis, width=mypdf.doc.width, height=400)
    mypdf.page_break()
    mypdf.p('Markers Distribution', 'h3')
    mypdf.img(info.marker_dis, width=mypdf.doc.width, height=400)
    mypdf.page_break()
    mypdf.p('Markers Co-Distribution', 'h3')
    mypdf.img(info.coexpr, width=mypdf.doc.width, height=600)

    mypdf.write_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--img_path', type=str, required=True, help="path for original img")
    parser.add_argument('--report_name', type=str, required=True, help="Output filepath")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()

    main(csv_path=args.csv_path, image_path=args.img_path, report_name=args.report_name, method=args.cluster_method)
