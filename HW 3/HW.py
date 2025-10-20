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
    # Create a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()
    
    # Replace '?' with NaN for missing values
    df_cleaned = df_cleaned.replace('?', np.nan)
    
    # Age: should be positive and reasonable (18-120)
    df_cleaned['Age'] = pd.to_numeric(df_cleaned['Age'], errors='coerce')
    df_cleaned.loc[(df_cleaned['Age'] < 18) | (df_cleaned['Age'] > 120), 'Age'] = np.nan
    
    # Sex: should be 0 or 1
    df_cleaned['Sex'] = pd.to_numeric(df_cleaned['Sex'], errors='coerce')
    df_cleaned.loc[~df_cleaned['Sex'].isin([0, 1]), 'Sex'] = np.nan
    
    # ChestPainType: should be 1, 2, 3, or 4
    df_cleaned['ChestPainType'] = pd.to_numeric(df_cleaned['ChestPainType'], errors='coerce')
    df_cleaned.loc[~df_cleaned['ChestPainType'].isin([1, 2, 3, 4]), 'ChestPainType'] = np.nan
    
    # RestBP: should be positive and reasonable (50-300 mmHg)
    df_cleaned['RestBP'] = pd.to_numeric(df_cleaned['RestBP'], errors='coerce')
    df_cleaned.loc[(df_cleaned['RestBP'] < 50) | (df_cleaned['RestBP'] > 300), 'RestBP'] = np.nan
    
    # Chol: should be positive and reasonable (100-600 mg/dl)
    df_cleaned['Chol'] = pd.to_numeric(df_cleaned['Chol'], errors='coerce')
    df_cleaned.loc[(df_cleaned['Chol'] < 100) | (df_cleaned['Chol'] > 600), 'Chol'] = np.nan
    
    # FBS: should be 0 or 1
    df_cleaned['FBS'] = pd.to_numeric(df_cleaned['FBS'], errors='coerce')
    df_cleaned.loc[~df_cleaned['FBS'].isin([0, 1]), 'FBS'] = np.nan
    
    # RestECG: should be 0, 1, or 2
    df_cleaned['RestECG'] = pd.to_numeric(df_cleaned['RestECG'], errors='coerce')
    df_cleaned.loc[~df_cleaned['RestECG'].isin([0, 1, 2]), 'RestECG'] = np.nan
    
    # MaxHR: should be positive and reasonable (60-220 bpm)
    df_cleaned['MaxHR'] = pd.to_numeric(df_cleaned['MaxHR'], errors='coerce')
    df_cleaned.loc[(df_cleaned['MaxHR'] < 60) | (df_cleaned['MaxHR'] > 220), 'MaxHR'] = np.nan
    
    # ExAng: should be 0 or 1
    df_cleaned['ExAng'] = pd.to_numeric(df_cleaned['ExAng'], errors='coerce')
    df_cleaned.loc[~df_cleaned['ExAng'].isin([0, 1]), 'ExAng'] = np.nan
    
    # Oldpeak: should be non-negative and reasonable (0-10)
    df_cleaned['Oldpeak'] = pd.to_numeric(df_cleaned['Oldpeak'], errors='coerce')
    df_cleaned.loc[(df_cleaned['Oldpeak'] < 0) | (df_cleaned['Oldpeak'] > 10), 'Oldpeak'] = np.nan
    
    # Slope: should be 1, 2, or 3
    df_cleaned['Slope'] = pd.to_numeric(df_cleaned['Slope'], errors='coerce')
    df_cleaned.loc[~df_cleaned['Slope'].isin([1, 2, 3]), 'Slope'] = np.nan
    
    # Ca: should be 0, 1, 2, or 3
    df_cleaned['Ca'] = pd.to_numeric(df_cleaned['Ca'], errors='coerce')
    df_cleaned.loc[~df_cleaned['Ca'].isin([0, 1, 2, 3]), 'Ca'] = np.nan
    
    # Thal: should be 3, 6, or 7
    df_cleaned['Thal'] = pd.to_numeric(df_cleaned['Thal'], errors='coerce')
    df_cleaned.loc[~df_cleaned['Thal'].isin([3, 6, 7]), 'Thal'] = np.nan
    
    # Num: should be 0, 1, 2, 3, or 4
    df_cleaned['Num'] = pd.to_numeric(df_cleaned['Num'], errors='coerce')
    df_cleaned.loc[~df_cleaned['Num'].isin([0, 1, 2, 3, 4]), 'Num'] = np.nan
    
    # Count missing values after cleaning
    missing_values_count = df_cleaned.isnull().sum().to_dict()
    
    return df_cleaned, missing_values_count


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
    # Create subsets with only categorical columns
    train_cat = df_train[categorical_columns].copy()
    val_cat = df_val[categorical_columns].copy()
    test_cat = df_test[categorical_columns].copy()
    
    # Initialize KNNImputer with k=5 and distance weights
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # Fit the imputer on training data and transform all datasets
    train_imputed = knn_imputer.fit_transform(train_cat)
    val_imputed = knn_imputer.transform(val_cat)
    test_imputed = knn_imputer.transform(test_cat)
    
    # Convert back to DataFrames with original indices
    train_imputed_df = pd.DataFrame(train_imputed, columns=categorical_columns, index=train_cat.index)
    val_imputed_df = pd.DataFrame(val_imputed, columns=categorical_columns, index=val_cat.index)
    test_imputed_df = pd.DataFrame(test_imputed, columns=categorical_columns, index=test_cat.index)
    
    # Round to nearest valid category value
    for col in categorical_columns:
        # Get valid values from training data
        valid_values = sorted(df_train[col].dropna().unique())
        
        # For each dataframe, round to nearest valid value
        for df in [train_imputed_df, val_imputed_df, test_imputed_df]:
            df[col] = df[col].apply(lambda x: min(valid_values, key=lambda y: abs(y - x)))
    
    return train_imputed_df, val_imputed_df, test_imputed_df


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
    # Make copies of numerical columns
    train_num = df_train[numerical_columns].copy()
    val_num = df_val[numerical_columns].copy()
    test_num = df_test[numerical_columns].copy()

    # Track which columns still have NaN
    cols_with_nan = train_num.columns[train_num.isna().any()].tolist()

    # Iterate until all missing values are filled
    while len(cols_with_nan) > 0:
        # Pick column with fewest missing values
        col_to_impute = train_num[cols_with_nan].isna().sum().idxmin()

        # Separate rows with and without missing values for the target column
        train_missing = train_num[train_num[col_to_impute].isna()]
        train_not_missing = train_num.dropna(subset=[col_to_impute])

        # Select feature columns: exclude target
        features = [c for c in numerical_columns if c != col_to_impute]

        # Keep only features without missing values in train_not_missing
        usable_features = [c for c in features if train_not_missing[c].isna().sum() == 0]

        if len(usable_features) == 0:
            # If no features are fully observed, fallback to median
            median_val = train_not_missing[col_to_impute].median()
            for df in [train_num, val_num, test_num]:
                df[col_to_impute].fillna(median_val, inplace=True)
        else:
            # Train Lasso on rows without NaN in usable features
            lasso = Lasso(alpha=0.1, random_state=8, max_iter=10000)
            lasso.fit(train_not_missing[usable_features], train_not_missing[col_to_impute])

            # Predict missing values in train, val, test
            for df in [train_num, val_num, test_num]:
                missing_rows = df[df[col_to_impute].isna()]
                if len(missing_rows) > 0:
                    predicted_values = lasso.predict(missing_rows[usable_features])
                    df.loc[missing_rows.index, col_to_impute] = predicted_values

        # Update columns with missing values
        cols_with_nan = train_num.columns[train_num.isna().any()].tolist()

    return train_num, val_num, test_num

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
    # Merge on index to combine categorical and numerical features
    merged_df = pd.concat([df_cat, df_num], axis=1)
    return merged_df


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
    # Set hyperparameters to Dictionary
    model.set_params(**hp)

    # Setting Categorical Transformer
    cat_transformer= OneHotEncoder(handle_unknown='ignore')

    # Setting Numerical Transformer
    num_transformer= StandardScaler()

    # Combining both Categorical and Numerical using Column Transformer
    preprocessor= ColumnTransformer(
        transformers=[
            ("cat",cat_transformer,cat_cols),
            ("num",num_transformer,num_cols)
    ])

    # Creating the Full Pipeline: Preprocessing + Model
    pipeline= Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Calling Fit Method for Training Data
    pipeline.fit(X_train, y_train)

    # Calling Predict Method for Validation
    y_pred = pipeline.predict(X_val)

    # Calculating F1 Score
    f1 = f1_score(y_val, y_pred)

    # Return 
    return {"params": hp, "F1 scores": f1}


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

    # Initializing F1 to calculate average later.
    f1_scores = []

    # Define the Spliter
    skf = StratifiedKFold(n_splits=cv, random_state=8, shuffle=True)

    # Loop and Split the Data into Train & Validation
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    
        # Scaling Categorical and Numerical Transformer
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
        num_transformer = StandardScaler()

        # Setting up Preprocessor using ColumnTransformer
        preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', cat_transformer, cat_cols),
                        ('num', num_transformer, num_cols)
                    ]
                )


        # Creating Pipeline
        pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model.set_params(**hp))
            ])
        
        # Calling Fit Method for Training Data
        pipeline.fit(X_train, y_train)

        # Calling Predict Method for Validation
        y_pred = pipeline.predict(X_val)

        # Calculating F1 Score
        f1 = f1_score(y_val, y_pred)

        # Appending f1 scores to average it later on.
        f1_scores.append(f1)

    # Computer Average of f1
    avg_f1 = np.mean(f1_scores)

    return {'params': hp, 'Average F1 scores': avg_f1}