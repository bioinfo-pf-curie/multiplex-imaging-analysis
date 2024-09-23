#!/usr/bin/env python

import argparse

def size_filter(cells, smin, smax):
    for cell in cells:
        if smin < cell.area < smax:
            yield cell

def segmented_area(mask):
    return mask.astype(bool).sum() / mask.size

def artefact_filter():
    pass

def co_expression_intracell():
    pass

def co_expression_neighboor():
    pass

def perform_filtering(img, size_min=None, size_max=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="original image path")
    parser.add_argument('--mask', type=str, required=True, help="mask path")
    parser.add_argument('--csv_path', type=str, required=True, help="path for csv file of quantification")
    parser.add_argument('--report_name', type=str, required=True, help="Output filepath")
    parser.add_argument('--cluster_method', type=str, required=False, default="phenograph", 
                        help="name of the cluster method (currently available : kmeans, phenograph or leiden)")
    args = parser.parse_args()

    perform_filtering(csv_path=args.csv_path, report_name=args.report_name, method=args.cluster_method)