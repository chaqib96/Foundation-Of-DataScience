from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

"""
HW.py

This file contains the skeleton for the functions you need to implement as part of your homework.
Each function corresponds to a specific task and includes instructions on what is expected.
"""

# Task 2: Data Cleaning
def clean_data(df):
    """
    Task: Data Cleaning
    --------------------
    This function should take a pandas DataFrame as input and return a cleaned DataFrame.
    
    Instructions:
    - Handle missing values in categorical and numerical columns separately.
    - Handle incorrect data points (e.g., negative or null weight values) (I know that there is no weight column!).
    - Ensure that in the cleaned dataframe all the missing or incorrect values are encoded as NaN.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    missing_values_count (dict): A dictionary with the count of missing values per column after cleaning.
    """
    pass


# Task 3: Categorical Features Imputation

def impute_missing_categorical(df_train, df_val, df_test, categorical_columns):
    """
    Task: Categorical Features Imputation
    --------------------------------------
    This function should handle missing values in categorical columns using appropriate techniques.
    We will skip scaling and encoding to keep things simple.
    
    Instructions:
    - Create subsets of the input DataFrames (train, validation, test) with only the categorical columns.
    - Use KNNImputer with k=5 and weights set to 'distance' to fill missing values in the categorical columns.
    - Ensure that the imputed values are approximated to the nearest value in the original dataset for each column, to avoid artifacts like decimal values.
    - If any "new value" is equidistant from two original values, choose the smaller one.
    - Add the column names to the resulting DataFrames after imputation.
    - The imputed dataframes should only contain the categorical columns.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_val (pd.DataFrame): The validation DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    categorical_columns (list): A list of column names corresponding to categorical features.

    Returns:
    pd.DataFrame: The training DataFrame with imputed categorical features.
    pd.DataFrame: The validation DataFrame with imputed categorical features.
    pd.DataFrame: The test DataFrame with imputed categorical features.
    """

    pass


# Task 4: Numerical Features Imputation
def impute_numerical_features(df_train, df_val, df_test, numerical_columns):
    """
    Task: Numerical Features Imputation
    ------------------------------------
    This function should handle missing values in numerical columns using appropriate techniques.
    Again, we will skip scaling to keep things simple.
    
    Instructions:
    - The function should take three datasets as input: df_train, df_val, and df_test.
    - The function should return three datasets as output: train_imputed_lasso, val_imputed_lasso, and test_imputed_lasso.
    - Use LassoRegressor in an iterative fashion to impute missing values in numerical columns.
    - Ensure that the imputation process is consistent and does not use any other imputer.
    - Follow these steps:
        1. Create a subset of the train dataset with only the numerical columns. Call this subset train_num.
        2. Create a subset of the val dataset with only the numerical columns. Call this subset val_num.
        3. Create a subset of the test dataset with only the numerical columns. Call this subset test_num.
        4a. Create a subset of train_num containing the rows with missing values. Call this subset train_num_missing.
        4b. Create a subset of train_num containing the rows without missing values. Call this subset train_num_not_missing.
        5a. Create a subset of val_num containing the rows with missing values. Call this subset val_num_missing.
        5b. Create a subset of val_num containing the rows without missing values. Call this subset val_num_not_missing.
        6a. Create a subset of test_num containing the rows with missing values. Call this subset test_num_missing.
        6b. Create a subset of test_num containing the rows without missing values. Call this subset test_num_not_missing.
        7a. Train a Lasso regression model on the correct subset (I am not telling you which one it is).
        7b. Using a Lasso regression, "predict" the missing values in the subsets that have missing values. Only predict the values in the column with the fewest missing values.
        8. Repeat steps 4-7 until all the missing values are imputed.
        9. Save the results in train_num_imputed_lasso, val_num_imputed_lasso, and test_num_imputed_lasso.
        10. Concatenate the imputed subsets with the subsets that did not contain missing values.
        11. Save the resulting datasets in train_imputed_lasso, val_imputed_lasso, and test_imputed_lasso.
        12. Ensure that the order of the rows in the final datasets matches the order in the original datasets.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_val (pd.DataFrame): The validation DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    numerical_columns (list): A list of column names corresponding to numerical features.

    Returns:
    pd.DataFrame: The training DataFrame with imputed numerical features.
    pd.DataFrame: The validation DataFrame with imputed numerical features.
    pd.DataFrame: The test DataFrame with imputed numerical features.
    """
    pass

def merge_imputed(df_cat, df_num):
    """
    Task: Merge Imputed DataFrames
    -------------------------------
    This function should merge the imputed categorical and numerical DataFrames.
    
    Instructions:
    - Merge the imputed categorical and numerical DataFrames on their indexes.
    - Ensure that the resulting DataFrame contains all columns from both input DataFrames.
    
    Parameters:
    df_cat (pd.DataFrame): The DataFrame with imputed categorical features.
    df_num (pd.DataFrame): The DataFrame with imputed numerical features.

    Returns:
    pd.DataFrame: The merged DataFrame containing both categorical and numerical features.
    """
    pass


# Task 5: Classification Using a Single Split
def train_and_evaluate_single_split(X_train, X_val, y_train, y_val, cat_cols, num_cols, model, hp):
    """
    Task: Classification Using a Single Split
    ------------------------------------------
    This function should train a classification pipeline on the training set and evaluate it on the validation set, using the provided parameters.

    Instructions:
    - Create a classification pipeline. It should include:
        - A OneHotEncoder for categorical features (handle_unknown='ignore').
        - A StandardScaler for numerical features.
        - The provided classification model.
    - Use ColumnTransformer to apply the appropriate transformations to categorical and numerical features.
    - Set the model parameters using the provided parameters dictionary.
    - Train the model using the training data (X_train, y_train).
    - Evaluate the model on the validation data (X_val, y_val) using F1 score.
    - Return the evaluation results (F1 score) for the given parameters combination.
    
    Parameters:
    X_train (pd.DataFrame): The training feature set.
    X_val (pd.DataFrame): The validation feature set.
    y_train (pd.Series): The training labels.
    y_val (pd.Series): The validation labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.

    Returns:
    dict: A dictionary containing two keys: 'params' (training parameters) and 'F1 scores' (F1 score). Each key should have the correct value.
    """
    pass


# Task 6: Classification Using Cross-Validation
def train_and_evaluate_cross_validation(X, y, model, cat_cols, num_cols, hp, cv):
    """
    Task: Classification Using Cross-Validation
    --------------------------------------------
    This function should train and evaluate a classification model using cross-validation.
    
    Instructions:
    - Use cross-validation to train and evaluate the model, with shuffle set to True and using the specified number of folds (cv).
    - For each fold, create a classification pipeline similar to the one in Task 5.
    - Evaluate the model on each fold using F1 score.
    - Return the average F1 score across all folds for each parameter combination.
    - Ensure that the cross-validation process is reproducible (use random_state = 8).
    
    Parameters:
    X (pd.DataFrame): The feature set.
    y (pd.Series): The labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.
    cv (int): The number of cross-validation folds.

    Returns:
    dict: A dictionary containing two keys: 'params' (training parameters) and 'Average F1 scores' (F1 score). Each key should have the correct value.
    """
    pass