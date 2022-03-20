#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 AUTHOR: Geert Oosterbroek
 DESCRIPTION:
 	Complete pipeline to go from count matrix to clusters for SEDEC
"""

import time
import pandas as pd
import sys
import numpy as np
from scipy import integrate
from numpy.linalg import norm
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon, Point
import multiprocessing
import hdbscan # To install
from read_roi import read_roi_zip
from datetime import date
import os
import pickle


def initialize():
    """
    Initialize the analysis with the given arguments
    :return: list with meta parameters
    """

    # Check number of utilized cores
    num_cpu = multiprocessing.cpu_count()
    print("Number of CPUs: ", num_cpu)

    # Check if using cluster or not
    if num_cpu > 8:
        isCluster = True
        from pandarallel import pandarallel
        pandarallel.initialize()
    else:
        isCluster = False

    arguments = sys.argv
    print("Parsed arguments", arguments)
    input_directory = arguments[1]
    output_directory = arguments[2]
    doSubset = arguments[3]
    return isCluster, input_directory, output_directory, doSubset


def subset(data, x_bounds=(3000,3500), y_bounds=(1000,1500)):
    """
    Subset the mRNA data to a subset with global defaults
    :param data:
    :param x_bounds: tuple indicating x-axis bounds
    :param y_bounds: tuple indicating y-axis bounds
    :return: Subsetted dataframe
    """
    return data.loc[(data['x'].between(x_bounds[0], x_bounds[1])) & (data['y'].between(y_bounds[0], y_bounds[1]))]


def calculate_bandwidth(data, scale_factor=1):
    """
    Calculate the bandwidth used for KDE as a fraction of the average cell size, based on nuclei ROIs
    :param data: ROI data
    :param scale_factor: fraction of average cell/nuclei size
    :return: bandwidth
    """

    total_size = 0

    for key in data.keys():
        xy = np.column_stack((data[key]['x'], data[key]['y']))
        poly = Polygon(xy)
        # get minimum bounding box around polygon
        box = poly.minimum_rotated_rectangle
        # get coordinates of polygon vertices
        x, y = box.exterior.coords.xy

        # get length of bounding box edges
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # get length of polygon as the longest edge of the bounding box
        length = max(edge_length)
        total_size += length

    average_size = total_size / len(data)
    gauss_fwtm_constant = 4.29
    bandwidth = (average_size / gauss_fwtm_constant) * scale_factor
    return bandwidth


def construct_kdes(data, bandwidth=20):
    """
    Constructs the kernel density estimator for each unique gene

    :param data: Data for which to construct KDEs, consisting out of spot locations and corresponding genes ('target')
    :return Series containing kde objects, one for each gene
    """

    grouped = data.groupby('target').agg(list)
    kdes = pd.Series(dtype=object)

    for target, coords in grouped.iterrows():
        # print(f"Working on KDE for gene: {target}")
        current_data = np.array(list(zip(*coords)))
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(current_data)
        kdes.loc[target] = kde
        # print(kde.get_params())
    return kdes


def density(x, y, kde):
    log_density= kde.score_samples(np.reshape([x, y], (1,-1)))
    density = np.exp(log_density)
    return density


def expression_vector(row, kdes, group_sizes):
    vec = pd.Series(dtype=object)
    for ind, kde in kdes.iteritems():
        # Integrate over pdf to obtain probability estimate
        probability = integrate.nquad(density, [[row['x']-0.5, row['x']+0.5], [row['y']-0.5, row['y']+0.5]], args=[kde])[0]
        vec.loc[ind] = float(probability * group_sizes.loc[ind])
    return vec


def cosine_similarity(a,b):
    return np.dot(a, b) / (norm(a) * norm(b))


def spex_distance(a, b):
    euclidean = norm(a[:2] - b[:2])  # First 2 elements are physical coordinates
    exp_vec_a, exp_vec_b = a[2:], b[2:]  # Elements thereafter are gene expressions
    cos_sim = np.dot(exp_vec_a, exp_vec_b) / (norm(exp_vec_a) * norm(exp_vec_b))
    if cos_sim > 0:
        return euclidean / cos_sim
    else:
        return np.inf


def main():
    print(sys.argv)
    isCluster, input_directory, output_directory, doSubset = initialize()
    df = pd.read_csv(input_directory + "507_s7 mRNA counts.csv")

    if doSubset == 'True':
        df = subset(df)

    cells = read_roi_zip(input_directory + "507_s7_all_cell_rois.zip")
    bandwidth = calculate_bandwidth(cells)
    print("Bandwidth: ", bandwidth)
    all_kdes = construct_kdes(df, bandwidth)
    group_size = df.groupby('target').size()

    t0 = time.time()

    if isCluster:
        print("starting with expression vectors")
        expression_vectors = df.parallel_apply(expression_vector, kdes=all_kdes, group_sizes=group_size, axis="columns")
    else:
        from tqdm import tqdm # check if works on cluster else use vanilla apply
        tqdm.pandas()
        expression_vectors = df.progress_apply(expression_vector, kdes=all_kdes, group_sizes=group_size, axis="columns")

    print(f"Time needed to calculate expression vectors: {time.time()-t0}")  # subset: 1570s, 26 min
    locations_expressions = df[['x', 'y']].join(expression_vectors)

    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(metric=spex_distance)
    clusterer.fit(locations_expressions)
    print(f"Time needed for clustering of subset: {time.time()-t0}")

    labels = clusterer.labels_
    hdb_cluster = pd.DataFrame(labels, columns=['cluster'])
    hdb_cluster.set_index(locations_expressions.index, inplace=True)
    locations_labels = pd.concat([df, hdb_cluster], axis=1)
    if doSubset:
        locations_labels.to_pickle(output_directory + f"{date.today()}-hdbscan_output_subset.pkl")
    else:
        locations_labels.to_pickle(output_directory + f"{date.today()}-hdbscan_output.pkl")


def assess_bandwidth():
    input_directory = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7\input"
    cell_rois = read_roi_zip(input_directory + r"\507_s7_all_cell_rois.zip")
    nuclei_rois = read_roi_zip(os.path.join(input_directory, "507_sc7_all_nuclei_RoiSet_crop.zip"))
    count_data = pd.read_csv(input_directory + r"\507_s7 mRNA counts.csv")
    count_data = subset(count_data)
    scale_factors = [0.1, 0.5, 0.8, 1, 1.5, 5, 10]
    labels = {}

    for factor in scale_factors:
        bandwidth = calculate_bandwidth(cell_rois, scale_factor=factor)
        print("Bandwidth: ", bandwidth)

        all_kdes = construct_kdes(count_data, bandwidth)
        group_size = count_data.groupby('target').size()
        from tqdm import tqdm # check if works on cluster else use vanilla apply
        tqdm.pandas()
        t0 = time.time()
        expression_vectors = count_data.progress_apply(expression_vector, kdes=all_kdes, group_sizes=group_size, axis="columns")

        print(f"Time needed to calculate expression vectors: {time.time()-t0}")  # subset: 1570s, 26 min
        locations_expressions = count_data[['x', 'y']].join(expression_vectors)

        clusterer = hdbscan.HDBSCAN(metric=spex_distance)
        clusterer.fit(locations_expressions)

        labels[f'{factor}'] = clusterer.labels_

    pickle.dump(labels, open(input_directory+"labels_different_bandwidth.pkl", "wb" ) )


if __name__ == '__main__':
    # main()
    assess_bandwidth()