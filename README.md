This file briefly describes the different source code files used to perform the analyses presented in this paper. The code itself can found in this Github repository. 

-  **data_prep.py** Module which takes in per gene count files from the CellProfiler output and returns a single count matrix. There each entry corresponds to an mRNA molecule, and gives its x-coordinate, y-coordinate and corresponding gene. Optionally, the coordinate changes which might have occurred because of the Matlab tiling can be undone. 
- **create_clustering.py** Module which takes in tabular data including the expression vector for each molecule (denoted as SPEX data in the code) and implements the SEDEC_clusterer class to run the SEDEC algorithm on this data. This file assumes the time-intensive calculation of the expression vectors has already been performed. 

- **integrated_clustering.py** Module to go from tabular count matrix to outcome clustering. Depending on arguments can be run locally or on a HPC. Can also be used to run the computationally demanding calculation of the expression vectors on a HPC and store these. 

- **expanded_nuclei.py** Implementation of the ExpandedNuclei class which runs the Expanded Nuclei algorithm on the tabular data and creates a clustering

- **nuclei_centroid.py** Implementation of the NCMap class which runs the Nuclei Centroid mapping algorithm on the tabular data and creates a clustering

- **helper_functions.py** File with functionalities which are re-used in several modules. 

- **exploration.ipynb** Jupyter notebook with initial data exploration and metrics

- **methodology.ipynb** Jupyter notebook to create plots used in the methodology section. Predominantly focuses on explaining the mechanisms behind HDBSCAN and providing a motivation for the SEDEC+ extension.  

- **results.ipynb** Jupyter notebook to create analyses of the results section.
