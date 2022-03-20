# -*- coding: utf-8 -*-
"""
Script to transform data into required format for further analyses.

Takes in per gene count file from CellProfiler, returns single count matrix.
Each entry corresponds to a mRNA molecule, and gives xCoord yCoord and corresponding gene.
Optionally, coordinate changes due to tiling can be undone.

@author: Geert Oosterbroek
"""

import pandas as pd
import numpy as np
import os
import re


def transform_coords(image_num, x_coord, y_coord, x_tile_size=2500, y_tile_size=2500):
    """
    Transform coordinates of a data point with respect to tiled image into coordinates with respect
    to complete image

    :param image_num: image number indicating tile position, starting from left bottom
    :param x_coord: x coordinate of data point
    :param y_coord: y coordinate of data point
    :param x_tile_size: width of image tile
    :param y_tile_size: height of image tile
    :return:
    """
    col = (image_num - 1) % 3
    row = (image_num - 1) // 3

    x = col * x_tile_size + x_coord
    y_t = row * y_tile_size + y_coord
    y = 5000-y_t
    return pd.Series({"x": x, "y": y})


def data_reader(count_location, target_location):
    """
    Read in files with all mRNAs per gene in tiled coordinates, return merged file with corrected coordinates
    :param count_location: path to directory storing count files per gene
    :param target_location: path to excel storing table with channel order - gene matching
    :return: dataframe containing all mRNA molecules' coordinates & corresponding gene
    """
    all_data = pd.DataFrame()
    target_table = pd.read_excel(target_location, nrows=20,
                                 usecols=["Channel Order", "Gene"], engine='openpyxl')

    for file in os.listdir(count_location):
        print(file)
        original_location_data = pd.read_csv(os.path.join(count_location, file),
                                             usecols=['ImageNumber', 'Location_Center_X', 'Location_Center_Y'])

        if original_location_data.empty:
            continue

        transformed_location_data = original_location_data.apply(
            lambda row: transform_coords(row['ImageNumber'], row['Location_Center_X'], row['Location_Center_Y']), 1)

        transformed_location_data["Channel Order"] = np.repeat(int(re.search(r'\d+', file).group(0)) + 1,
                                                               len(transformed_location_data))
        transformed_location_data = pd.merge(transformed_location_data, target_table, left_on='Channel Order',
                                             right_on='Channel Order', how='left')
        all_data = all_data.append(transformed_location_data)

    all_data.rename(columns={'Gene': 'target'}, inplace=True)
    all_data.drop(columns=['Channel Order'], inplace=True)
    return all_data


def main():
    count_per_gene_folder = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7\Counts per gene"
    channel_order_file = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7\channel_order.xlsx"
    store_path = r"C:\Users\geertoosterbroek\Documents\Thesis\Data\507_s7\507_s7 mRNA counts final.csv"

    # Check if there is already a complete dataset in place, in which case we do not overwrite
    if os.path.exists(store_path):
        print("Data file already exists on this location. Please check!")
    else:
        count_data = data_reader(count_per_gene_folder, channel_order_file)
        count_data.to_csv(store_path, index=False)

if __name__ == '__main__':
    main()
