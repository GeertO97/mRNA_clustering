"""
 AUTHOR: Geert Oosterbroek
 DESCRIPTION:
 	Script to obtain SEDEC clusters from SPEX data
"""
import numpy as np
import pandas as pd
import hdbscan
import os
import sys

from read_roi import read_roi_zip
from scipy.cluster.vq import kmeans2
sys.path.insert(1, r'C:\Users\geertoosterbroek\Documents\Thesis\Code\mRNA-2-nuclei\code')
from helper_functions import subset, spex_distance, find_nuclei_centroids


def _find_closest_centroid(point, selected_centroids):
    """
    For particular mRNA molecule find closest centroid in dictionary
    """
    closest_key = -1
    closest_distance = 10000
    for key, value in selected_centroids.items():
        distance = np.linalg.norm(point[['x', 'y']] - value)
        if distance < closest_distance:
            closest_distance = distance
            closest_key = key
    return closest_key


def _label_closest_nuclei(group, centroids):
    """
    Find label of closest nuclei for all mRNA molecules
    """

    if (group['sedec_labels'] == -1).all():
        # Molecules classified as noise should remain this way
        labels = [-1] * len(group)
    else:
        # For molecules not regarded as noise find closest nucleus
        _, labels = kmeans2(group[['x', 'y']], k=centroids, iter=1)
        unique, counts = np.unique(labels, return_counts=True)

        # Only centroids with at least m_pts molecules can remain
        sufficient_clusters = unique[counts >= 5]
        selected_centroids = dict(zip(sufficient_clusters, centroids[sufficient_clusters,:]))

        # create boolean values of labels that do not belong to selected centroids
        needs_reassignment_mask = [True if i not in sufficient_clusters else False for i in labels]
        if sum(needs_reassignment_mask) > 0:
            labels[needs_reassignment_mask] = group.loc[needs_reassignment_mask].apply(
                _find_closest_centroid, axis='columns', args=(selected_centroids,))

    group['closest_nuclei_label'] = labels

    return group


class SEDEC_clusterer:

    def __init__(self, min_pts=10, min_clsize=5):
        self.spex_data = pd.DataFrame()
        self.min_pts = min_pts
        self.min_clsize = min_clsize
        self.labels = []
        self.labels_plus = []
        self.output_data = pd.DataFrame()
        self.centroids = None

    def fit(self, spex_data):
        self.spex_data = spex_data
        self.output_data = self.spex_data.copy()
        sedec = hdbscan.HDBSCAN(metric=spex_distance, gen_min_span_tree=True, min_samples=self.min_pts,
                                min_cluster_size=self.min_clsize).fit(self.spex_data)
        self.labels = sedec.labels_
        self.output_data['sedec_labels'] = self.labels
        return self

    def sedec_plus(self, nuclei_rois):
        self.centroids = find_nuclei_centroids(nuclei_rois)
        sedec_plus_dataset = self.output_data.groupby('sedec_labels', as_index=False).apply(
            _label_closest_nuclei, centroids=self.centroids)

        # Find sedec plus labels as unique combinations of sedec labels and nuclei labels
        sedec_plus_dataset['sedec_plus_labels'] = sedec_plus_dataset.\
            groupby(['sedec_labels','closest_nuclei_label']).ngroup()
        sedec_plus_dataset.loc[sedec_plus_dataset['sedec_plus_labels'] == 0, 'sedec_plus_labels'] = -1
        self.output_data = sedec_plus_dataset
        return self


def main():
    do_subset = True
    data_dir = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7"
    spex_data = pd.read_pickle(os.path.join(data_dir, "input", "exp_vec_full.pkl"))

    if do_subset:
        spex_data = subset(spex_data)

    clusterer = SEDEC_clusterer()
    clusterer.fit(spex_data)

    # Implementation of sedec+
    nuclei_rois = read_roi_zip(os.path.join(data_dir, "input", "507_sc7_all_nuclei_RoiSet_crop.zip"))

    clusterer.sedec_plus(nuclei_rois)

    # Store labelled data in appropriate folders
    output_dir = os.path.join(data_dir, "output")
    if do_subset:
        output_dir = os.path.join(output_dir, "subset")

    # pickle.dump(clusterer, open(os.path.join(output_dir, "SEDEC_fitted.pkl"), "wb"))
    clusterer.output_data.to_csv(os.path.join(output_dir, "SEDEC_labelled_data.csv"))


if __name__ == '__main__':
    main()
