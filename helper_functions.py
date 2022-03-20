"""
 AUTHOR: Geert Oosterbroek
 DESCRIPTION:
 	Helper functions which can be used in multiple scripts
"""
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def subset(data, x_bounds=(3000,3500), y_bounds=(1000,1500)):
    """
    Subset the mRNA data to a subset with global defaults
    :param data:
    :param x_bounds:
    :param y_bounds:
    :return:
    """
    return data.loc[(data['x'].between(x_bounds[0], x_bounds[1])) & (data['y'].between(y_bounds[0], y_bounds[1]))]


def spex_distance(a, b):
    """
    Calculate SPatial-EXpressional distance between two observations,
    defined by the Euclidean distance of their coordinates divided by the cosine similarity of their expression
    :param a:
    :param b:
    :return:
    """
    euclidean = norm(a[:2] - b[:2])  # First 2 elements are physical coordinates
    exp_vec_a, exp_vec_b = a[2:], b[2:]  # Elements thereafter are gene expressions
    cos_sim = np.dot(exp_vec_a, exp_vec_b) / (norm(exp_vec_a) * norm(exp_vec_b))
    if cos_sim > 0:
        return euclidean / cos_sim
    else:
        return np.inf


def find_nuclei_centroids(nuclei):
    """
    Function to retrieve the nuclei centroids
    :param nuclei:
    :return:
    """
    center_points = np.empty((len(nuclei), 2))

    for ind, key in enumerate(nuclei.keys()):
        x, y = nuclei[key]['x'], nuclei[key]['y']
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        center_points[ind,:] = (x_mean, y_mean)

    return center_points


def calculate_ari_ami_matrix(label_df, metric):
    """
    Calculate upper triangular matrix of Adjusted Rand Index or Adjusted Mutual Information
    values between columns of label_df which should contain the labels corresponding to a cluster outcome per column
    """

    metric = metric.lower()
    scores = np.zeros((label_df.shape[1],label_df.shape[1]))

    for i in range(label_df.shape[1]):
        for j in range(i, label_df.shape[1]):
            if metric=='ari':
                scores[i,j] = adjusted_rand_score(label_df.iloc[:,i], label_df.iloc[:,j])
            elif metric=='ami':
                scores[i,j] = adjusted_mutual_info_score(label_df.iloc[:,i], label_df.iloc[:,j])
            else:
                raise ValueError("Choose either ari or ami metric")


    return scores
