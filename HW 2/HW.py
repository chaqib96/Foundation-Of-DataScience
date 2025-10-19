from __future__ import annotations

from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# task_1.py



def load_data(path: str) -> pd.DataFrame:
    """
    Read the CSV dataset "Cust_Segmentation.csv" and return a pandas DataFrame.

    TODO:
      - Use pandas to read the CSV dataset "Cust_Segmentation.csv".
      - Return the DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    # TODO: implement
    raise NotImplementedError("Implement load_data()")


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Drop categorical columns, drop rows with NaN, keep numeric columns, and standardize.

    Steps:
      1) Drop columns: "Customer Id", "Defaulted", "Address"  (ignore if missing)
      2) Keep only numeric columns
      3) Drop rows containing NaNs
      4) Standardize remaining features with StandardScaler (mean≈0, std≈1)
      5) Return (X_scaled, feature_names, df_clean)

    Returns
    -------
    X_scaled : np.ndarray        # shape (n_samples, n_features), standardized
    feature_names : List[str]    # names of the numeric features used
    df_clean : pd.DataFrame      # cleaned numeric DataFrame (pre-standardization)

    Notes
    -----
    The tests expect the banned columns to be removed and no NaNs in X.
    """
    # TODO: implement:
    #   drop_cols = ["Customer Id", "Defaulted", "Address"]
    #   df_clean = ...
    #   feature_names = ...
    #   X_scaled = StandardScaler().fit_transform(...)
    raise NotImplementedError("Implement preprocess(df)")
    
    
# task_2.py


def elbow_inertia(X: np.ndarray, k_min: int = 1, k_max: int = 10, random_state: int = 42) -> List[float]:
    """
    Fit KMeans for k in [k_min..k_max] and return the list of inertias.

    TODO:
      - Loop k from k_min to k_max (inclusive)
      - Fit KMeans(n_clusters=k, random_state=random_state)
      - Append km.inertia_ to a list
      - Return the list of inertias (length should be k_max - k_min + 1)
    """
    # TODO: implement
    raise NotImplementedError("Implement elbow_inertia(X, k_min, k_max, random_state)")


def identify_elbow_k() -> int:
    """
    Identify the 'elbow' k from a list of inertias.
    
    TODO:
      - Return the chosen k (int)

    Notes
    -----
    The tests will assert the number you return for this dataset.
    """
    # TODO: implement heuristic and return k for this dataset
    raise NotImplementedError("Implement identify_elbow_k()")
    
    
    
# task_3.py



def kmeans_cluster(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Fit KMeans and return (labels, fitted_model).

    TODO:
      - Initialize KMeans with given n_clusters/random_state
      - labels = km.fit_predict(X)
      - Return (labels, km)
    """
    # TODO: implement
    raise NotImplementedError("Implement kmeans_cluster(X, n_clusters, random_state)")


def kmeans_add_labels_and_centroids(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add K-Means labels to df_clean and compute centroids per cluster.

    Output:
      - df_with_labels: df_clean + a new column 'cluster_kmeans'
      - centroids_df:   DataFrame of per-cluster means for the columns in feature_names
                        (rows indexed by cluster id, columns = feature_names)

    TODO:
      - Make a copy of df_clean
      - Add column 'cluster_kmeans' with the given labels
      - Group by 'cluster_kmeans' and compute mean over feature_names
      - Return (df_with_labels, centroids_df)
    """
    # TODO: implement
    raise NotImplementedError("Implement kmeans_add_labels_and_centroids(...)")
    

# task_4.py



def _count_clusters(labels: np.ndarray) -> int:
    """Helper: number of clusters excluding noise (-1)."""
    uniq = set(labels.tolist())
    if -1 in uniq:
        uniq.remove(-1)
    return len(uniq)


def dbscan_cluster_to_target_k(
    X: np.ndarray,
    target_k: int = 1,
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Try a small grid of (eps, min_samples) values until you reach exactly target_k clusters
    (excluding noise). Return (labels, fitted_model).

    TODO:
      - Define a small grid, e.g. eps in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0], min_samples in [3, 5, 8]
      - For each pair, fit DBSCAN and check number of clusters via _count_clusters
      - If exactly target_k, return (labels, model). target_k should be equal to the elbow point in Task 2.
      - If none match, return the best/last attempt
    """
    # TODO: implement
    raise NotImplementedError("Implement dbscan_cluster_to_target_k(X, target_k)")


def dbscan_add_labels(df_clean: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Return a copy of df_clean with a new column 'cluster_dbscan' containing the labels.

    TODO:
      - Copy df_clean
      - Add column 'cluster_dbscan' with the provided labels
      - Return the new DataFrame
    """
    # TODO: implement
    raise NotImplementedError("Implement dbscan_add_labels(df_clean, labels)")



# task_5.py



def compute_silhouettes(
    X: np.ndarray,
    km_labels: np.ndarray,
    db_labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute silhouette scores for K-Means and DBSCAN.

    Notes:
      - K-Means: use all samples
      - DBSCAN: ignore noise points (label == -1). Only compute if >=2 clusters remain.

    TODO:
      - km_sil = silhouette_score(X, km_labels)
      - For DBSCAN:
          mask out noise; if at least 2 clusters remain and >=2 samples, compute score on masked data
          else set db_sil = np.nan
      - Return (km_sil, db_sil) as floats
    """
    # TODO: implement
    raise NotImplementedError("Implement compute_silhouettes(X, km_labels, db_labels)")



