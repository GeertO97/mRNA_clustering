#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 AUTHOR: Geert Oosterbroek
 DESCRIPTION: 
 	Expanded nuclei approach to mRNA clustering,
 	current default method
"""

from read_roi import read_roi_zip
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys
import pickle
sys.path.insert(1, r'C:\Users\geertoosterbroek\Documents\Thesis\Code\mRNA-2-nuclei\code')
from helper_functions import subset


class ExpandedNuclei:
    """
    Implementation of mRNA clustering based on expanding pre determined nuclei ROIs
    """

    def __init__(self):
        self.count_data = None
        self.cell_rois = None
        self.labels = None
        self.output_data = None

    def fit(self, count_data, cell_rois):
        """
        Perform clustering of mRNA molecules based on expanded nuclei ROIs
        :parameters: self, instance of class
        :return: self, fitted instance of class including cluster labels
        """

        self.count_data = count_data
        self.output_data = self.count_data.copy()
        self.cell_rois = cell_rois
        # By default points are classified as noise
        labels = np.full(len(self.count_data), -1)

        for ind, key in enumerate(self.cell_rois.keys()):
            x, y = self.cell_rois[key]['x'], self.cell_rois[key]['y']
            xy = np.column_stack((x, y))
            poly = patches.Polygon(xy)  # We first make a polygon to get a closed path
            path = poly.get_path()
            # Find which observations are in the closed path, i.e. inside the ROI region
            inpath_mask = path.contains_points(count_data[['x', 'y']])
            labels[inpath_mask] = ind

        self.labels = labels
        self.output_data['exp_nuc_labels'] = labels
        return self


def main():
    do_subset = True
    data_dir = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7"
    input_directory = os.path.join(data_dir, "input")
    count_data = pd.read_csv(os.path.join(input_directory, "507_s7 mRNA counts.csv"))
    cell_rois = read_roi_zip(os.path.join(input_directory, "507_s7_all_cell_rois.zip"))
    if do_subset:
        count_data = subset(count_data)
    expanded_nuclei = ExpandedNuclei().fit(count_data, cell_rois)

    # Store labelled data in appropriate folders
    output_dir = os.path.join(data_dir, "output")
    if do_subset:
        output_dir = os.path.join(output_dir, "subset")

    # pickle.dump(clusterer, open(os.path.join(output_dir, "SEDEC_fitted.pkl"), "wb"))
    expanded_nuclei.output_data.to_csv(os.path.join(output_dir, "exp_nuc_labelled_data.csv"))


if __name__ == '__main__':
    main()

