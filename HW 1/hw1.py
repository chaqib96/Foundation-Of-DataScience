"""
HW1 — Foundations of Data Science
Theme: Data preprocessing, EDA, PCA
Dataset: hotel_bookings.csv

SUBMISSION NOTES
- Implement functions exactly as named.
- Do NOT rename columns, and do NOT drop duplicates (beyond Task 1 rules).
- Expected cleaned shape on the official dataset: (119210, 33).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================ CONSTANTS ============================
# Exact feature list for PCA. Do NOT change order or names.
PCA_FEATURES: List[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_guests",
]


# ============================ TASK 1 ============================
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Task 1 — Load & Light Clean (simple instructions)

    What you will do:
    - Turn the reservation status date into a real date type.
    - Turn obvious numeric columns into real numbers (bad values become NaN).
    - Create a new column total_guests = adults + children + babies.
    - Remove rows where total_guests == 0 (bookings must have guests).
    - If adr (average daily rate) is negative, set it to NaN (do not drop the row).

    Why this matters:
    These steps make sure dates are real dates, numbers are real numbers,
    bookings make sense (no zero guests), and errors are marked as missing values.

    Expected shape on the official dataset:
    - (119210, 33) after cleaning.

    Return:
    - DataFrame with all original columns PLUS 'total_guests'.
    - Do NOT drop duplicates. Do NOT fill other missing values here.
    """

    # TODO 1 — Read the CSV file into a DataFrame
    
    # TODO 2 — Convert reservation_status_date into a proper datetime (invalid -> NaT)
    
    # TODO 3 — Convert obvious numeric columns to numbers (invalid -> NaN)
    numeric_cols = [
        "is_canceled", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "days_in_waiting_list", "adr",
        "required_car_parking_spaces", "total_of_special_requests",
        "arrival_date_year", "arrival_date_day_of_month",
    ]
    
    # TODO 4 — Create total_guests = adults + children + babies (row-wise sum)
    
    # TODO 5 — Keep only rows where total_guests > 0
    
    # TODO 6 — Handle negative prices: if adr < 0, set it to NaN (do NOT drop the row)
    
    # TODO 7 — Return the cleaned DataFrame
   
    
# ============================ TASK 2 ============================
def numeric_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Task 2 — Basic Numeric KPIs (simple instructions)

    What you will do:
    - Take the cleaned DataFrame from Task 1.
    - Compute a few key numbers (no plotting/printing).
    - Return them in a small dictionary.

    KPIs to compute:
    - rows          -> number of rows in the cleaned data
    - cols          -> number of columns
    - cancel_rate   -> mean of is_canceled
    - adr_p95       -> 95th percentile of adr
    - avg_stay_len  -> mean of total nights
                       (stays_in_week_nights + stays_in_weekend_nights)
    """

    # TODO 1 — rows

    # TODO 2 — cols

    # TODO 3 — cancel_rate

    # TODO 4 — adr_p95

    # TODO 5 — avg_stay_len

    # TODO 6 — package & return
    

# ============================ TASK 3 ============================
def categorical_cancel_stats(df: pd.DataFrame) -> Dict[str, object]:
    """
    Task 3 — Categorical EDA (simple instructions)

    What you will do:
    - Use the cleaned DataFrame from Task 1.
    - Compute cancellation rates (mean of is_canceled) by:
        * hotel        -> hotel_rates (dict)
        * deposit_type -> deposit_rates (dict)
    - Find the market segment with the highest cancel rate among
      segments with at least 500 rows -> ('segment', rate_float).
    """

    # TODO 1 — hotel_rates

    # TODO 2 — deposit_rates

    # TODO 3 — top_segment_500
    
    # TODO 4 — return dict
    

# ============================ TASK 4 ============================
def build_pca_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Task 4 — Build PCA-ready Matrix (simple instructions)

    What you will do:
    - Use the cleaned DataFrame from Task 1.
    - Select EXACTLY the 14 numeric features in PCA_FEATURES (in order).
    - Median-impute missing values, then standardize (z-score).
    - Return a NumPy array with shape (n_samples, 14), no NaNs.
    """

    # TODO 1 — select columns in order
    
    # TODO 2 — median imputation
    
    # TODO 3 — standardize
    
    # TODO 4 — return array
    

# ============================ TASK 5 ============================
def run_pca(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 5 — Run PCA (simple instructions)

    What you will do:
    - Take the PCA-ready matrix X from Task 4 (no NaNs, standardized; shape: (n_samples, 14)).
    - Fit PCA with n_components=3.
    - Return TWO arrays:
        1) explained_variance_ratio_ -> shape (3,)
        2) components_               -> shape (3, 14)
    """

    # TODO 1 — init PCA

    # TODO 2 — fit

    # TODO 3 — grab arrays
    
    # TODO 4 — return
   
# ============================ END OF FILE ============================