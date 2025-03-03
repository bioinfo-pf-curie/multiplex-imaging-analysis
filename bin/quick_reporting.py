#!/usr/bin/env python

import os
import argparse
# import scimap as sm
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import tifffile
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from jinja2 import Template
import json

from utils import min_max_norm
from quality_control import SIZE_MAX, SIZE_MIN, NECROTIC, AOI_IN, AOI_OUT

# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.platypus import SimpleDocTemplate, Image, Table, Paragraph, Spacer, PageBreak


pio.templates.default = "plotly_white"


# class ScimapGraph:
#     def __init__(self, csv_path, method):
#         self.df = self.prepare_data(csv_path)
#         self.compute_stuff(method)

#         self.spatial_colormap(self.df, color_by=method, filepath="scimap/test_colormap.jpg")
#         self.spatial_interaction(self.df, filepath="scimap/test_interaction.jpg")

#         sm.pl.spatial_interaction(self.df, 
#                           spatial_interaction='spatial_interaction_radius',
#                           linewidths=0.75, linecolor='black', figsize=(5,4), saveDir="scimap", fileName="spatial_interaction.jpg")
#         sm.pl.heatmap(self.df, groupBy=method, standardScale="column",
#                     saveDir="scimap", fileName="cluster_phenotype.jpg")

#         sm.pl.voronoi(self.df, color_by='spatial_aggregate_radius', 
#                     voronoi_edge_color = 'black',
#                     voronoi_line_width = 0.3, 
#                     voronoi_alpha = 0.8, 
#                     size_max=3000,
#                     overlay_points=None,
#                     saveDir="scimap", fileName="voronoi.jpg",
#                     legend_size=6)
    
#     @staticmethod
#     def prepare_data(csv_path):
#         return sm.pp.mcmicro_to_scimap(csv_path)
    
#     def compute_stuff(self, method):
#         self.unsupervised_clustering(method)
#         self.df = sm.tl.spatial_interaction(self.df, phenotype=method,
#                                     method='radius', 
#                                     radius=70, 
#                                     label='spatial_interaction_radius')
#         self.df = sm.tl.spatial_aggregate(self.df, phenotype=method, 
#                                         method='radius', radius=50, purity=80, label='spatial_aggregate_radius')

    
#     @staticmethod
#     def spatial_interaction(adata, spatial_interaction='spatial_interaction_radius', filepath="."):
#         filepath = Path(filepath)
#         return sm.pl.spatialInteractionNetwork(adata, spatial_interaction=spatial_interaction, figsize=(6,4), saveDir=filepath.parent, fileName=filepath.name)
    
#     def unsupervised_clustering(self, method='leiden'):
#         self.df = sm.tl.cluster(self.df, method=method, use_raw=False, log=False)

#     @staticmethod
#     def spatial_colormap(adata, color_by=['leiden_phenotype'], filepath=".", **kwargs):
#         filepath = Path(filepath)
#         return sm.pl.spatial_scatterPlot (adata, colorBy=color_by, s=3, 
#                                         fontsize=5, catCmap='Set1', saveDir=filepath.parent, 
#                                         fileName=filepath.name, **kwargs)

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

    def __init__(self, img_path, csv_path, parms):
        self.df = pd.read_csv(csv_path)
        self.nb_cell = len(self.df)
        
        self.area_min = self.get_number_filtered_cells(SIZE_MIN)
        self.area_max = self.get_number_filtered_cells(SIZE_MAX)
        self.necro = self.get_number_filtered_cells(NECROTIC)
        self.roi = self.get_number_filtered_cells(AOI_IN)
        self.exclu = self.get_number_filtered_cells(AOI_OUT)

        self.marker_cols = [col for col in self.df if col not in self.cn.values()]

        self.parms = parms
        self.mask = None

        self.tissue_fraction = np.nan

        # self.size_dis = self.make_size_distribution(height=650)
        # self.marker_dis = self.make_markers_distribution(height=650)
        # coexpr_size = max(600, 70 * len(self.marker_cols))
        # self.coexpr = self.make_co_expression(height=coexpr_size, width=coexpr_size)

        self.tiff = tifffile.TiffFile(img_path)
        if self.tiff.series[0].is_pyramidal:
            self.thumbnail = self.tiff.series[0].levels[-1].asarray()
        else: 
            total_size = 1
            for dim in self.tiff.series[0].shape:
                total_size *= dim
            self.thumbnail = self.tiff.series[0].asarray() if total_size * 2 / (1024 * 1024) < 200 else None # total size < 200 Mo

        if self.thumbnail is not None:
            i,a = np.quantile(self.thumbnail, [0.01,0.99])
            self.thumbnail = min_max_norm(self.thumbnail, i, a, output_max=255)
            self.mask = np.zeros(self.thumbnail.shape[1:], dtype="int32")

        if self.parms.get('ROIPath', False): 
            self.make_mask(self.parms['ROIPath'], 1)
        if self.parms.get('excludedPath', False): 
            self.make_mask(self.parms['excludedPath'], 2)
        try:
            self.segmented_fraction = self.get_fraction_segmented()
        except ValueError:
            self.segmented_fraction = np.nan

    def make_mask(self, geojson, color):
        import cv2
        if self.mask is None:
            return None
        coords = self.read_geojson(geojson)
        for poly in coords:
            aa = np.array(poly[0], dtype=np.int32).reshape(-1,1,2)
            self.mask = cv2.polylines(self.mask, [aa], 1, color)

    def get_number_filtered_cells(self, col_name):
        col_mins = [col for col in self.df if col_name in col]
        if col_mins:
            self.cn[col_name] = col_mins[0]
            return self.df[self.cn[col_name]].count()
        return 0
    
    def read_geojson(self, geojson_path):
        with open(geojson_path, 'r') as gjfile:
            gj = json.load(gjfile)

        if gj['type'] == "FeatureCollection":
            features = gj['features']
        elif gj['type'] == 'Feature':
            features = [gj]
        else:
            raise ValueError(f'Unrecognize type in geojson {geojson_path}')
        
        return [roi['geometry']['coordinates'] for roi in features]


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
    
    def heatmap_data_co_expr(self):
        return self.df[self.marker_cols].corr()


# class PDFReport:

#     def __init__(self, reportpath, pagesize=A4):
#         self.styles = getSampleStyleSheet()
#         self.path = reportpath
#         self.doc = SimpleDocTemplate(self.path, pagesize=pagesize)
#         self.doc_width, self.doc_height = pagesize
#         self.parts = []

#     def p(self, text, style='BodyText'):
#         self.parts.append(Paragraph(text, style=self.styles[style]))

#     def header(self, title):
#         self.p(title, 'h1')

#     def spacer(self, height=100, width=None):
#         if width is None:
#             width = self.doc_width
#         self.parts.append(Spacer(width=width, height=height))

#     def img(self, src, width=None, height=None):
#         img = Image(src, width=width, height=height)
#         img.vAlign = "MIDDLE"
#         self.parts.append(img)

#     def page_break(self):
#         self.parts.append(PageBreak())

#     def write_report(self):
#         # parts.append(Image("scimap/spatial_interaction.jpg", width=400, height=560))
#         # parts.append(Image("scimap/test_colormap.jpg", width=400, height=560))
#         # parts.append(Image("scimap/test_interaction.jpg", width=400, height=560))
#         # parts.append(Image("scimap/voronoi.jpg", width=400, height=560))
#         self.doc.build(self.parts)

def tiff2rgb(img, geom=None, out_path="thumbnail.png"):
    color_cycle = [[int(h.strip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)] 
                   for h in ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']]
    result = np.transpose(np.stack([img[0], img[1], img[2]]), (1,2,0)) # take the first three channels as RGB

    # and merge the rest
    for channel in range(3, img.shape[0]):
        tmp_img = np.transpose(np.stack([img[channel]] * 3) * np.array(color_cycle[channel % len(color_cycle)])[:,None,None], (1,2,0))
        alpha = 1 / (channel + 1)
        result = cv2.addWeighted(result, 1-alpha, tmp_img, alpha, 0, dtype=cv2.CV_8UC1)

    if geom is not None:
        result[geom == 1] = (79,244,255) # inclusion in yellow (bgr)
        result[geom == 2] = (44,30,240) # exclusion in red (bgr)

    cv2.imwrite(str(out_path), result.astype('uint8'))
    return out_path

def plot_violin(data):
    ignored_cols = ['CellID', "X_centroid", "Y_centroid"]
    cols = [col for col in data.columns if col not in ignored_cols]
    fig = make_subplots(rows=len(cols), cols=1, vertical_spacing=0)
    for i, col in enumerate(cols, 1):
        fig.add_trace(go.Violin(
            x=data[col], orientation="h", name=col
        ), row=i, col=1)
    fig.update_layout(height=100 * len(cols), template="plotly_white")
    return fig

def plot_box(data):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    ignored_cols = ['CellID', "X_centroid", "Y_centroid"]
    cols = [col for col in data.columns if col not in ignored_cols]
    box_parms = data[cols].quantile([.25,.5,.75]).T
    iqr = (box_parms[.75] - box_parms[.25]) * 1.5
    box_parms['lf'], box_parms['uf'] = box_parms[.25] - iqr, box_parms[.75] + iqr
    fig = make_subplots(rows=len(cols), cols=1, vertical_spacing=0)
    for i, col in enumerate(cols, 1):
        p = box_parms.loc[col]
        lf = data[col] - p['lf']
        lf.loc[lf < 0] = np.nan
        try:
            lf = data.iloc[lf.idxmin()][col]
        except:
            lf = p[.25]
        uf = data[col] - p['uf']
        uf.loc[uf > 0] = np.nan
        try:
            uf = data.iloc[uf.idxmax()][col]
        except:
            uf = p[.75]
        fig.add_trace(go.Box(
            q1=[p[.25]], median=[p[.5]], q3=[p[.75]], lowerfence=[lf], upperfence=[uf], orientation="h", name=col, marker_color=colors[i%len(colors)],
            y0=col
        ), row=i, col=1)
        fig.add_trace(go.Box(
            x=data.loc[(data[col] < lf) | (data[col] > uf), col], 
            boxpoints="all", fillcolor='rgba(255,255,255,0)', line={'color': 'rgba(255,255,255,0)'}, marker_color=colors[i%len(colors)],
            showlegend=False, hoveron='points', pointpos=0, hovertemplate='x=%{x}<extra></extra>', y0=col
        ), row=i, col=1)
    fig.update_layout(height=100 * len(cols), template="plotly_white")
    return fig


def main(image_path, csv_path, parms, out_dir):
    df_gen = {}
    for img, csv in zip(image_path, csv_path):
        info = GetBasicInfo(img, csv, parms)
        img_name = Path(img).stem

        # write gen stat
        df_gen[img_name] = {'Cell number': info.nb_cell,
                            "Tissue / Background area": info.tissue_fraction * 100,
                            "Segmented Fraction": info.segmented_fraction * 100, 
                            "Cells number under minimal size": info.area_min, 
                            "Cells number over maximal size": info.area_max, 
                            "Necrotics cells": info.necro, 
                            "Cells in Region of Interest": info.roi, "Cells in excluded region": info.exclu}

        # create violin plot
        fig = plot_box(info.df)
        fig.write_html(out_dir / f"{img_name}_markers_distribution_mqc.html", full_html=False, include_plotlyjs=False)

        # create thumbnail
        tiff2rgb(info.thumbnail, info.mask, out_path= out_dir / f"{img_name}_thumbnail.png")
        
        # write heatmap data
        info.heatmap_data_co_expr().to_csv(out_dir / f'{img_name}_heatmap.csv')

        # write methods
        with open(Path(os.environ.get('NXF_ASSETS')) / "method_template.html", 'r') as templatef:
            template = Template(templatef.read())

        with open(out_dir / f'{img_name}_methods_mqc.html', 'w') as out:
            out.write(template.render())
        
    df_gen = pd.DataFrame.from_dict(df_gen, orient='index')
    df_gen.index.name = "Image name"
    df_gen.to_csv(out_dir / 'report_stats.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, nargs="+", required=True, help="path for csv file of quantification")
    parser.add_argument('--img_path', type=str, nargs="+", required=True, help="path for original img")
    parser.add_argument('--out_dir', type=str, required=True, help="Output filepath")
    parser.add_argument('--parms', type=json.loads, required=False, help="parameters used")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()
    main(csv_path=args.csv_path, image_path=args.img_path, 
         out_dir=Path(args.out_dir), parms=args.parms)
