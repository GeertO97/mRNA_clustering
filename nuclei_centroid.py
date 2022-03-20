import pickle
import numpy as np
from read_roi import read_roi_zip
import pandas as pd
from scipy.cluster.vq import kmeans2
import sys
import os
sys.path.insert(1, r'C:\Users\geertoosterbroek\Documents\Thesis\Code\mRNA-2-nuclei\code')
from helper_functions import subset, find_nuclei_centroids


class NCMap:
    """
    Implementation of Nuclei Centroid Mapping,
    based on the notion of Voronoi Tessellation to cluster mRNA molecules
    """

    def __init__(self):
        self.count_data = None
        self.nuclei_rois = None
        self.centroids = None
        self.labels = None
        self.output_data = None

    def fit(self, count_data, nuclei_rois):
        """
        Perform clustering of mRNA molecules based on voronoi tessellation
        :parameters: self, instance of class
        :return: self, fitted instance of class including cluster labels
        """

        self.count_data = count_data
        self.output_data = self.count_data.copy()
        self.nuclei_rois = nuclei_rois
        self.centroids = find_nuclei_centroids(self.nuclei_rois)
        _, self.labels = kmeans2(self.count_data[['x', 'y']], k=self.centroids, iter=1)
        self.post_process()
        self.output_data['nc_labels'] = self.labels
        return self

    def post_process(self, min_size=5):
        """
        Remove too small clusters, as these are likely noise
        :parameters:
            self, instance of class
            min_size, minimum cluster size parameter
        :return: self, post-processed instance
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        sufficient_clusters = unique[counts >= min_size]
        self.labels = np.asarray([i if i in sufficient_clusters else -1 for i in self.labels])
        return self


def main():
    do_subset = True
    data_dir = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7"
    input_directory = os.path.join(data_dir, "input") #"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7\input"
    count_data = pd.read_csv(os.path.join(input_directory, "507_s7 mRNA counts.csv"))
    nuclei_rois = read_roi_zip(os.path.join(input_directory, "507_sc7_all_nuclei_RoiSet_crop.zip"))
    if do_subset:
        count_data = subset(count_data)
    nuclei_mapping = NCMap().fit(count_data, nuclei_rois)

    # Store labelled data in appropriate folders
    output_dir = os.path.join(data_dir, "output")
    if do_subset:
        output_dir = os.path.join(output_dir, "subset")

    nuclei_mapping.output_data.to_csv(os.path.join(output_dir, "nucmap_labelled_data.csv"))


if __name__ == '__main__':
    main()
