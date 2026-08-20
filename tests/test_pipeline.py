import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROCESSED_DATA_DIR = "data/processed"
MODEL_PATH = "models/random_forest.pkl"


# ============================================================
# TEST 1 — PROCESSED DATA DIRECTORY
# ============================================================

def test_processed_data_directory_exists():
    """Verify that the processed-data directory exists."""
    assert os.path.isdir(PROCESSED_DATA_DIR)


# ============================================================
# TEST 2 — CLEANED SALES DATA
# ============================================================

def test_sales_cleaned_file():
    """Verify that the cleaned sales dataset exists and loads."""
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "sales_cleaned.csv"
    )

    assert os.path.isfile(path)

    df = pd.read_csv(path)

    assert not df.empty


# ============================================================
# TEST 3 — REQUIRED SALES COLUMNS
# ============================================================

def test_sales_required_columns():
    """Verify that important sales columns exist."""
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "sales_cleaned.csv"
    )

    df = pd.read_csv(path)

    required_columns = {
        "Date",
        "Product ID",
        "Units Sold",
        "Demand"
    }

    assert required_columns.issubset(df.columns)


# ============================================================
# TEST 4 — FINAL MODEL
# ============================================================

def test_final_model_exists():
    """Verify that the trained Random Forest model exists."""
    assert os.path.isfile(MODEL_PATH)


# ============================================================
# TEST 5 — MODEL COMPARISON
# ============================================================

def test_model_comparison_file():
    """Verify that model-comparison results exist."""
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "model_comparison.csv"
    )

    assert os.path.isfile(path)

    df = pd.read_csv(path)

    assert not df.empty


# ============================================================
# TEST 6 — FUTURE FORECAST
# ============================================================

def test_future_forecast_file():
    """Verify that future demand forecasts exist."""
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "future_demand_forecast.csv"
    )

    assert os.path.isfile(path)

    df = pd.read_csv(path)

    assert not df.empty


# ============================================================
# TEST 7 — INVENTORY ANALYSIS
# ============================================================

def test_inventory_analysis_file():
    """Verify that inventory-analysis results exist."""
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "inventory_analysis.csv"
    )

    assert os.path.isfile(path)

    df = pd.read_csv(path)

    assert not df.empty