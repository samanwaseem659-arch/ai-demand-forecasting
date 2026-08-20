
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

/* ================================
   GLOBAL
================================ */

.stApp {
    background: #F5F7FB;
    color: #172033;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] {
    background: #172033;
    min-width: 245px;
    max-width: 245px;
}

section[data-testid="stSidebar"] > div {
    background: #172033;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar navigation buttons */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border: none;
    background: transparent;
    color: #D0D5DD !important;
    text-align: left;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 5px;
    font-size: 14px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #26344D;
    color: #FFFFFF !important;
}

/* ================================
   HEADINGS
================================ */

.main-title {
    font-size: 30px;
    font-weight: 750;
    color: #172033;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 14px;
    color: #667085;
    margin-bottom: 24px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #172033;
    margin-top: 25px;
    margin-bottom: 14px;
}

/* ================================
   KPI CARDS
================================ */

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E6EAF0;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}

.kpi-label {
    font-size: 13px;
    color: #667085;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 27px;
    font-weight: 750;
    color: #172033;
}

.kpi-description {
    font-size: 12px;
    color: #98A2B3;
    margin-top: 6px;
}

/* ================================
   BUSINESS CARDS
================================ */

.business-card {
    background: #FFFFFF;
    border: 1px solid #E6EAF0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    margin-bottom: 20px;
}

.card-title {
    font-size: 17px;
    font-weight: 700;
    color: #172033;
    margin-bottom: 4px;
}

.card-subtitle {
    font-size: 13px;
    color: #667085;
    margin-bottom: 16px;
}

/* ================================
   STATUS BADGES
================================ */

.status-normal {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #ECFDF3;
    color: #027A48;
    font-size: 12px;
    font-weight: 600;
}

.status-warning {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #FFFAEB;
    color: #B54708;
    font-size: 12px;
    font-weight: 600;
}

.status-danger {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #FEF3F2;
    color: #B42318;
    font-size: 12px;
    font-weight: 600;
}

/* ================================
   BUTTONS
================================ */

.stButton > button {
    border-radius: 8px;
    border: 1px solid #D0D5DD;
    background: #FFFFFF;
    color: #172033;
    font-weight: 600;
    min-height: 40px;
}

.stButton > button:hover {
    border-color: #172033;
    background: #F8FAFC;
}

/* ================================
   DOWNLOAD BUTTON
================================ */

.stDownloadButton > button {
    background: #172033;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: #26344D;
    color: #FFFFFF;
}

/* ================================
   INPUTS
================================ */

.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div {
    border-radius: 8px;
}

/* ================================
   TABLE
================================ */

.dataframe {
    border-radius: 10px;
}

/* ================================
   FILE UPLOADER
================================ */

section[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF;
    border: 1px dashed #98A2B3;
    border-radius: 12px;
}

/* ================================
   DIVIDER
================================ */

hr {
    border-color: #E6EAF0;
}

/* ================================
   MOBILE
================================ */

@media (max-width: 900px) {

    .main-title {
        font-size: 24px;
    }

    .kpi-card {
        margin-bottom: 12px;
    }

}


</style>
""", unsafe_allow_html=True)


# ============================================================
# APPLICATION TITLE
# ============================================================



# ============================================================
# PATHS
# ============================================================

from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="Samannnnww/ai-demand-forecasting",
    filename="random_forest.pkl"
)

@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception:
        return None
DATA_PATH = "data/raw/sales_data.csv"
MODEL_COMPARISON_PATH = "data/processed/model_comparison.csv"


# ============================================================
# LOAD SAVED AI MODEL
# ============================================================

@st.cache_resource
def load_ai_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = joblib.load(MODEL_PATH)
        return model

    except Exception:
        return None


ai_model = load_ai_model()


# ============================================================
# REQUIRED DATA SCHEMA
# ============================================================

REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
    "Demand"
]


NUMERIC_COLUMNS = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Demand"
]


# ============================================================
# DATA DICTIONARY
# ============================================================

DATA_DICTIONARY = {

    "Date": (
        "Date of the sales/inventory observation.",
        "Used for time-series ordering, trend analysis and forecasting."
    ),

    "Store ID": (
        "Unique store identifier.",
        "Used to analyze demand and inventory at store level."
    ),

    "Product ID": (
        "Unique product identifier.",
        "Used to generate product-level forecasts."
    ),

    "Category": (
        "Product category.",
        "Used for category-level demand analysis."
    ),

    "Region": (
        "Geographical/business region.",
        "Used to compare demand between regions."
    ),

    "Inventory Level": (
        "Current recorded inventory.",
        "Used for stockout, overstock and reorder analysis."
    ),

    "Units Sold": (
        "Historical quantity sold.",
        "Provides historical sales information."
    ),

    "Units Ordered": (
        "Quantity ordered historically.",
        "Helps understand replenishment behavior."
    ),

    "Price": (
        "Product selling price.",
        "Business variable that may influence demand."
    ),

    "Discount": (
        "Discount applied to the product.",
        "Used to study promotional/price effects."
    ),

    "Weather Condition": (
        "Recorded weather condition.",
        "External variable that may influence demand."
    ),

    "Promotion": (
        "Promotion indicator/value.",
        "Used to analyze promotional demand effects."
    ),

    "Competitor Pricing": (
        "Competitor product price.",
        "Useful for competitive pricing analysis."
    ),

    "Seasonality": (
        "Season or seasonal indicator.",
        "Used to detect seasonal demand patterns."
    ),

    "Epidemic": (
        "Epidemic/event indicator.",
        "Represents an external event that may affect demand."
    ),

    "Demand": (
        "Historical demand value.",
        "Primary target variable for demand forecasting."
    )
}


# ============================================================
# CSV READER
# ============================================================

@st.cache_data
def read_csv(path, date_col=None):

    df = pd.read_csv(path)

    if date_col and date_col in df.columns:

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col
        for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    return missing_columns, extra_columns


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(df):

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    invalid_dates = int(
        data["Date"].isna().sum()
    )

    data = data.dropna(
        subset=["Date"]
    ).copy()

    for col in NUMERIC_COLUMNS:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    numeric_missing_before = int(
        data[NUMERIC_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    data[NUMERIC_COLUMNS] = (
        data[NUMERIC_COLUMNS]
        .fillna(0)
    )

    text_columns = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    for col in text_columns:

        data[col] = (
            data[col]
            .fillna("Unknown")
            .astype(str)
        )

    for col in [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Demand"
    ]:

        data[col] = (
            data[col]
            .clip(lower=0)
        )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    cleaning_report = {
        "Invalid Date Rows Removed": invalid_dates,
        "Numeric Missing Values Filled": numeric_missing_before,
        "Rows After Cleaning": len(data),
        "Columns": len(data.columns)
    }

    return data, cleaning_report


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data
def engineer_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------------

    data["DayOfWeek"] = (
        data["Date"].dt.dayofweek
    )

    data["DayOfMonth"] = (
        data["Date"].dt.day
    )

    data["Month"] = (
        data["Date"].dt.month
    )

    data["Quarter"] = (
        data["Date"].dt.quarter
    )

    data["Year"] = (
        data["Date"].dt.year
    )

    data["WeekOfYear"] = (
        data["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["IsWeekend"] = (
        data["DayOfWeek"] >= 5
    ).astype(int)

    data["IsHoliday"] = (
        data["Date"]
        .dt.dayofyear
        .isin([1, 14, 23, 25, 26])
        .astype(int)
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    data = (
        data
        .sort_values(
            [
                "Store ID",
                "Product ID",
                "Date"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # GROUPS
    # --------------------------------------------------------

    grouped_demand = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
    )

    grouped_price = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Price"]
    )

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    data["Demand Lag 1"] = (
        grouped_demand.shift(1)
    )

    data["Demand Lag 7"] = (
        grouped_demand.shift(7)
    )

    data["Demand Lag 14"] = (
        grouped_demand.shift(14)
    )

    data["Demand Lag 30"] = (
        grouped_demand.shift(30)
    )

    # --------------------------------------------------------
    # ROLLING MEANS
    # --------------------------------------------------------

    data["Demand Rolling Mean 7"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .mean()
        )
    )

    data["Demand Rolling Mean 14"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                14,
                min_periods=1
            )
            .mean()
        )
    )

    data["Demand Rolling Mean 30"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                30,
                min_periods=1
            )
            .mean()
        )
    )

    # --------------------------------------------------------
    # ROLLING STANDARD DEVIATION
    # --------------------------------------------------------

    data["Demand Rolling Std 7"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=2
            )
            .std()
        )
    )

    data["Demand Rolling Std 14"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                14,
                min_periods=2
            )
            .std()
        )
    )

    # --------------------------------------------------------
    # MIN / MAX
    # --------------------------------------------------------

    data["Demand Rolling Min 7"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .min()
        )
    )

    data["Demand Rolling Max 7"] = (
        data
        .groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .max()
        )
    )

    # --------------------------------------------------------
    # HISTORICAL DEMAND
    # --------------------------------------------------------

    data["Historical Demand"] = (
        data["Demand"].shift(1)
    )

    # --------------------------------------------------------
    # PRICE FEATURES
    # --------------------------------------------------------

    data["Previous Price"] = (
        grouped_price.shift(1)
    )

    data["Price Change"] = (
        data["Price"] -
        data["Previous Price"]
    )

    # IMPORTANT:
    # Exact feature name expected by trained model
    data["Price_Change"] = (
        data["Price Change"]
    )

    data["Price Change %"] = (
        (
            data["Price"] -
            data["Previous Price"]
        )
        /
        data["Previous Price"].replace(
            0,
            np.nan
        )
        * 100
    )

    # --------------------------------------------------------
    # PROMOTION
    # --------------------------------------------------------

    data["Promotion Indicator"] = (
        pd.to_numeric(
            data["Promotion"],
            errors="coerce"
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # CATEGORY / REGION
    # --------------------------------------------------------

    data["Product Category"] = (
        data["Category"].astype(str)
    )

    data["Store Region"] = (
        data["Region"].astype(str)
    )

    # --------------------------------------------------------
    # DISCOUNT
    # --------------------------------------------------------

    data["Discount Feature"] = (
        pd.to_numeric(
            data["Discount"],
            errors="coerce"
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # COMPETITOR PRICE
    # --------------------------------------------------------

    data["Competitor Price Difference"] = (
        data["Price"] -
        data["Competitor Pricing"]
    )

    # --------------------------------------------------------
    # CLEAN ENGINEERED FEATURES
    # --------------------------------------------------------

    numeric_features = [

        "Demand Lag 1",
        "Demand Lag 7",
        "Demand Lag 14",
        "Demand Lag 30",

        "Demand Rolling Mean 7",
        "Demand Rolling Mean 14",
        "Demand Rolling Mean 30",

        "Demand Rolling Std 7",
        "Demand Rolling Std 14",

        "Demand Rolling Min 7",
        "Demand Rolling Max 7",

        "Historical Demand",

        "Previous Price",
        "Price Change",
        "Price_Change",
        "Price Change %",

        "Promotion Indicator",
        "Discount Feature",

        "Competitor Price Difference"
    ]

    data[numeric_features] = (
        data[numeric_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return data


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(series, fallback=0.0):

    value = pd.to_numeric(
        series,
        errors="coerce"
    ).mean()

    if pd.notna(value):
        return float(value)

    return float(fallback)


# ============================================================
# AI MODEL PREDICTION
# ============================================================

def predict_with_ai_model(feature_row):

    if ai_model is None:
        return None

    try:

        # ----------------------------------------------------
        # Find exact model features
        # ----------------------------------------------------

        required_features = None

        if hasattr(
            ai_model,
            "feature_names_in_"
        ):

            required_features = list(
                ai_model.feature_names_in_
            )

        elif hasattr(
            ai_model,
            "named_steps"
        ):

            for step in ai_model.named_steps.values():

                if hasattr(
                    step,
                    "feature_names_in_"
                ):

                    required_features = list(
                        step.feature_names_in_
                    )

                    break

        # ----------------------------------------------------
        # If model does not expose names,
        # use current data
        # ----------------------------------------------------

        if required_features is None:

            return None

        # ----------------------------------------------------
        # Create missing model features
        # ----------------------------------------------------

        for col in required_features:

            if col not in feature_row.columns:

                feature_row[col] = 0

        # ----------------------------------------------------
        # Exact order
        # ----------------------------------------------------

        model_input = feature_row[
            required_features
        ].copy()

        # ----------------------------------------------------
        # Convert possible object columns
        # ----------------------------------------------------

        for col in model_input.columns:

            if model_input[col].dtype == "object":

                model_input[col] = (
                    model_input[col]
                    .astype(str)
                )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = ai_model.predict(
            model_input
        )[0]

        return max(
            0,
            float(prediction)
        )

    except Exception as exc:

        return None


# ============================================================
# FORECAST + INVENTORY ENGINE
# ============================================================

@st.cache_data(show_spinner="Calculating demand and inventory insights...")
def generate_dynamic_results(
    df,
    horizon=7
):

    data = engineer_features(df)

    last_date = data["Date"].max()

    forecast_rows = []
    inventory_rows = []
    trend_rows = []

    for (
        store,
        product
    ), group in data.groupby(
        [
            "Store ID",
            "Product ID"
        ],
        sort=True
    ):

        g = group.sort_values(
            "Date"
        ).copy()

        category = str(
            g["Category"].iloc[-1]
        )

        region = str(
            g["Region"].iloc[-1]
        )

        current_inventory = float(
            g["Inventory Level"].iloc[-1]
        )

        # ----------------------------------------------------
        # RECENT DEMAND
        # ----------------------------------------------------

        recent = g.tail(
            min(14, len(g))
        )

        previous = g.iloc[
            max(0, len(g) - 28):
            max(0, len(g) - 14)
        ]

        recent_mean = safe_mean(
            recent["Demand"]
        )

        previous_mean = safe_mean(
            previous["Demand"],
            recent_mean
        )

        overall_mean = safe_mean(
            g["Demand"],
            recent_mean
        )

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if previous_mean > 0:

            change_pct = (
                (
                    recent_mean -
                    previous_mean
                )
                /
                previous_mean
                * 100
            )

            trend_factor = float(
                np.clip(
                    recent_mean /
                    previous_mean,
                    0.85,
                    1.15
                )
            )

        else:

            change_pct = 0.0
            trend_factor = 1.0

        if change_pct > 5:

            trend_name = "Increasing"

        elif change_pct < -5:

            trend_name = "Decreasing"

        else:

            trend_name = "Stable"

        # ----------------------------------------------------
        # WEEKDAY PATTERN
        # ----------------------------------------------------

        weekday_means = (
            g.groupby(
                "DayOfWeek"
            )["Demand"]
            .mean()
        )

        # ----------------------------------------------------
        # DEMAND VARIABILITY
        # ----------------------------------------------------

        recent_std = (
            g["Demand"]
            .tail(
                min(30, len(g))
            )
            .std()
            if len(g) > 1
            else 0
        )

        if pd.isna(recent_std):

            recent_std = 0

        recent_std = float(
            recent_std
        )

        # ----------------------------------------------------
        # SAFETY STOCK
        # ----------------------------------------------------

        safety_stock = max(
            0,
            1.65 * recent_std
        )

        if safety_stock == 0:

            safety_stock = (
                recent_mean * 0.20
            )

        # ----------------------------------------------------
        # FUTURE DATES
        # ----------------------------------------------------

        future_dates = pd.date_range(
            last_date +
            pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        predictions = []

        # ----------------------------------------------------
        # FUTURE PREDICTIONS
        # ----------------------------------------------------

        for future_date in future_dates:

            weekday = (
                future_date.dayofweek
            )

            weekday_average = float(
                weekday_means.get(
                    weekday,
                    overall_mean
                )
            )

            if overall_mean > 0:

                weekday_factor = (
                    weekday_average /
                    overall_mean
                )

            else:

                weekday_factor = 1

            weekday_factor = float(
                np.clip(
                    weekday_factor,
                    0.75,
                    1.25
                )
            )

            # ------------------------------------------------
            # CREATE FUTURE MODEL ROW
            # ------------------------------------------------

            future_row = g.iloc[
                -1:
            ].copy()

            future_row["Date"] = (
                future_date
            )

            future_row["DayOfWeek"] = (
                future_date.dayofweek
            )

            future_row["DayOfMonth"] = (
                future_date.day
            )

            future_row["Month"] = (
                future_date.month
            )

            future_row["Quarter"] = (
                future_date.quarter
            )

            future_row["Year"] = (
                future_date.year
            )

            future_row["WeekOfYear"] = int(
                future_date.isocalendar().week
            )

            future_row["IsWeekend"] = int(
                future_date.dayofweek >= 5
            )

            # ------------------------------------------------
            # Make sure model-required features exist
            # ------------------------------------------------

            future_row["Price_Change"] = (
                future_row["Price"] -
                future_row["Previous Price"]
            )

            future_row["Price Change"] = (
                future_row["Price_Change"]
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            ai_prediction = (
                predict_with_ai_model(
                    future_row
                )
            )

            if ai_prediction is not None:

                prediction = ai_prediction

            else:

                # ------------------------------------------------
                # FALLBACK FORECAST
                # ------------------------------------------------

                prediction = max(
                    0,
                    recent_mean *
                    trend_factor *
                    weekday_factor
                )

            predictions.append(
                prediction
            )

            forecast_rows.append(
                {
                    "Date": future_date,
                    "Store ID": store,
                    "Product ID": product,
                    "Category": category,
                    "Region": region,
                    "Predicted Demand": round(
                        prediction,
                        2
                    )
                }
            )

        # ----------------------------------------------------
        # INVENTORY PLANNING
        # ----------------------------------------------------

        average_daily_demand = (
            float(
                np.mean(predictions)
            )
            if predictions
            else recent_mean
        )

        forecast_demand = float(
            np.sum(predictions)
        )

        required_inventory = (
            forecast_demand +
            safety_stock
        )

        lead_time_days = 2

        reorder_level = (
            average_daily_demand *
            lead_time_days +
            safety_stock
        )

        reorder_quantity = max(
            0,
            reorder_level -
            current_inventory
        )

        if average_daily_demand > 0:

            coverage_days = (
                current_inventory /
                average_daily_demand
            )

        else:

            coverage_days = np.inf

        # ----------------------------------------------------
        # INVENTORY STATUS
        # ----------------------------------------------------

        if current_inventory <= reorder_level:

            status = "STOCKOUT RISK"

            risk_level = "High"

            action = (
                "Reorder immediately to "
                "protect against stockout."
            )

        elif current_inventory < required_inventory:

            status = "LOW STOCK"

            risk_level = "Medium"

            action = (
                "Increase inventory toward "
                "the recommended level."
            )

        elif current_inventory > (
            required_inventory * 1.5
        ):

            status = "OVERSTOCK"

            risk_level = "Medium"

            action = (
                "Reduce new orders and "
                "consider inventory redistribution."
            )

        else:

            status = "NORMAL"

            risk_level = "Low"

            action = (
                "Inventory level is appropriate "
                "for forecasted demand."
            )

        # ----------------------------------------------------
        # INVENTORY RESULT
        # ----------------------------------------------------

        inventory_rows.append(
            {
                "Date": last_date,
                "Store ID": store,
                "Product ID": product,
                "Category": category,
                "Region": region,
                "Inventory Level": round(
                    current_inventory,
                    2
                ),
                "Predicted Demand": round(
                    forecast_demand,
                    2
                ),
                "Average Daily Demand": round(
                    average_daily_demand,
                    2
                ),
                "Inventory Coverage Days": (
                    round(
                        float(
                            coverage_days
                        ),
                        2
                    )
                    if np.isfinite(
                        coverage_days
                    )
                    else 999
                ),
                "Safety Stock": round(
                    safety_stock,
                    2
                ),
                "Reorder Level": round(
                    reorder_level,
                    2
                ),
                "Required Inventory": round(
                    required_inventory,
                    2
                ),
                "Stock Difference": round(
                    current_inventory -
                    required_inventory,
                    2
                ),
                "Inventory Status": status,
                "Risk Level": risk_level,
                "Recommended Reorder Quantity": round(
                    reorder_quantity,
                    2
                ),
                "Recommended Action": action
            }
        )

        # ----------------------------------------------------
        # TREND RESULT
        # ----------------------------------------------------

        trend_rows.append(
            {
                "Store ID": store,
                "Product ID": product,
                "Category": category,
                "Demand Trend": trend_name,
                "Previous Average Demand": round(
                    previous_mean,
                    2
                ),
                "Recent Average Demand": round(
                    recent_mean,
                    2
                ),
                "Demand Change %": round(
                    change_pct,
                    2
                )
            }
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    forecast = pd.DataFrame(
        forecast_rows
    )

    inventory = pd.DataFrame(
        inventory_rows
    )

    trend = pd.DataFrame(
        trend_rows
    )

    # ========================================================
    # WEEKLY PATTERN
    # ========================================================

    weekly = (
        data
        .groupby(
            "DayOfWeek",
            as_index=False
        )["Demand"]
        .mean()
        .rename(
            columns={
                "Demand":
                "Average Demand"
            }
        )
    )

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekly["Day"] = (
        weekly["DayOfWeek"]
        .map(
            lambda x:
            day_names[int(x)]
        )
    )

    # ========================================================
    # MONTHLY PATTERN
    # ========================================================

    monthly = (
        data
        .groupby(
            "Month",
            as_index=False
        )["Demand"]
        .mean()
        .rename(
            columns={
                "Demand":
                "Average Demand"
            }
        )
    )

    monthly["MonthName"] = (
        monthly["Month"]
        .map(
            lambda x:
            pd.Timestamp(
                2000,
                int(x),
                1
            ).strftime("%B")
        )
    )

    # ========================================================
    # SEASONAL PATTERN
    # ========================================================

    seasonal = (
        data
        .groupby(
            "Seasonality",
            as_index=False
        )["Demand"]
        .mean()
        .rename(
            columns={
                "Demand":
                "Average Demand"
            }
        )
    )

    return (
        forecast,
        inventory,
        trend,
        weekly,
        monthly,
        seasonal,
        data
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔎 Control Panel"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="font-size:1.15rem;font-weight:700;">
        📊 AI Demand Intelligence
    </div>

    <div style="font-size:.8rem;opacity:.75;margin-top:4px;">
        Forecasting • Inventory • Insights
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📤 Dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload a CSV containing the required demand forecasting columns."
)

uploaded_df = None
cleaning_report = {}


if uploaded_file is not None:

    try:

        candidate = pd.read_csv(
            uploaded_file
        )

        missing_columns, extra_columns = (
            validate_dataset(candidate)
        )

        if missing_columns:

            st.sidebar.error(
                "❌ Dataset validation failed."
            )

            st.sidebar.write(
                "Missing columns:"
            )

            for col in missing_columns:

                st.sidebar.write(
                    f"• {col}"
                )

        else:

            uploaded_df, cleaning_report = (
                clean_dataset(
                    candidate
                )
            )

            st.session_state[
                "uploaded_data"
            ] = uploaded_df

            st.session_state[
                "cleaning_report"
            ] = cleaning_report

            st.sidebar.success(
                "✅ Dataset validated"
            )

            st.sidebar.info(
                f"Rows: {len(uploaded_df):,}"
            )

            st.sidebar.info(
                f"Columns: {len(uploaded_df.columns)}"
            )

            if extra_columns:

                st.sidebar.caption(
                    "Additional columns detected: "
                    +
                    ", ".join(
                        extra_columns[:8]
                    )
                )

    except Exception as exc:

        st.sidebar.error(
            f"Dataset error: {exc}"
        )


# ============================================================
# KEEP DATA THROUGH RERUNS
# ============================================================

if uploaded_df is None:

    if "uploaded_data" in st.session_state:

        uploaded_df = (
            st.session_state[
                "uploaded_data"
            ].copy()
        )

        cleaning_report = (
            st.session_state.get(
                "cleaning_report",
                {}
            )
        )


# ============================================================
# FORECAST SETTINGS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📅 Forecast Settings"
)

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [
        "Next 7 Days",
        "Next 14 Days",
        "Next 30 Days",
        "Custom"
    ],
    key="forecast_horizon"
)


if forecast_horizon == "Next 7 Days":

    forecast_days = 7

elif forecast_horizon == "Next 14 Days":

    forecast_days = 14

elif forecast_horizon == "Next 30 Days":

    forecast_days = 30

else:

    forecast_days = st.sidebar.number_input(
        "Number of Forecast Days",
        min_value=1,
        max_value=90,
        value=60,
        step=1
    )


forecast_days = int(
    forecast_days
)


# ============================================================
# DEFAULT DATASET
# ============================================================

if uploaded_df is not None:

    active_data = uploaded_df.copy()

    data_mode = "Uploaded Dataset"

else:

    try:

        active_data = read_csv(
            DATA_PATH,
            "Date"
        )

        active_data, cleaning_report = (
            clean_dataset(
                active_data
            )
        )

        data_mode = "Project Dataset"

    except Exception as exc:

        st.error(
            "Unable to load the project dataset."
        )

        st.error(
            str(exc)
        )

        st.stop()


# ============================================================
# GENERATE RESULTS
# ============================================================

try:

    (
        forecast,
        inventory,
        trend,
        weekly_pattern,
        monthly_pattern,
        seasonal_pattern,
        engineered_data
    ) = generate_dynamic_results(
        active_data,
        forecast_days
    )

except Exception as exc:

    st.error(
        f"Forecasting engine error: {exc}"
    )

    st.stop()


# ============================================================
# MODEL COMPARISON
# ============================================================

try:

    model_comparison = read_csv(
        MODEL_COMPARISON_PATH
    )

except Exception:

    model_comparison = pd.DataFrame()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "⚙️ System Status"
)

st.sidebar.success(
    "Dataset loaded"
)

st.sidebar.success(
    "Data validation active"
)

st.sidebar.success(
    "Feature engineering active"
)

if ai_model is not None:

    st.sidebar.success(
        "AI Random Forest model loaded"
    )

else:

    st.sidebar.warning(
        "AI model unavailable"
    )

st.sidebar.success(
    "Forecast engine active"
)

st.sidebar.success(
    "Inventory analysis active"
)


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📑 Application Sections"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Exploratory Data Analysis",
        "📈 Demand Forecast",
        "📦 Inventory Intelligence",
        "🔎 Demand Patterns",
        "🧠 AI Business Insights",
        "📚 Data & Workflow",
        "🤖 Model Performance",
        "⚙️ System Validation"
    ],
    label_visibility="collapsed"
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎯 Product Filters"
)

stores = sorted(
    forecast["Store ID"]
    .dropna()
    .unique()
    .tolist()
)

if not stores:

    st.error(
        "No stores found."
    )

    st.stop()


selected_store = st.sidebar.selectbox(
    "Store",
    stores
)


store_forecast = forecast[
    forecast["Store ID"] ==
    selected_store
].copy()


products = sorted(
    store_forecast["Product ID"]
    .dropna()
    .unique()
    .tolist()
)


if not products:

    st.error(
        "No products found for selected store."
    )

    st.stop()


selected_product = st.sidebar.selectbox(
    "Product",
    products
)


# ============================================================
# SELECTED DATA
# ============================================================

product_forecast = (
    store_forecast[
        store_forecast["Product ID"] ==
        selected_product
    ]
    .sort_values("Date")
    .head(forecast_days)
)


product_inventory = inventory[
    (
        inventory["Store ID"] ==
        selected_store
    )
    &
    (
        inventory["Product ID"] ==
        selected_product
    )
].copy()


product_trend = trend[
    (
        trend["Store ID"] ==
        selected_store
    )
    &
    (
        trend["Product ID"] ==
        selected_product
    )
].copy()


# ============================================================
# GLOBAL KPIs
# ============================================================

total_products = (
    forecast[
        [
            "Store ID",
            "Product ID"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


total_inventory = (
    inventory[
        "Inventory Level"
    ].sum()
)


total_forecast = (
    forecast[
        "Predicted Demand"
    ].sum()
)


stockout_count = int(
    inventory[
        "Inventory Status"
    ]
    .eq("STOCKOUT RISK")
    .sum()
)


overstock_count = int(
    inventory[
        "Inventory Status"
    ]
    .eq("OVERSTOCK")
    .sum()
)


increasing_count = int(
    trend[
        "Demand Trend"
    ]
    .eq("Increasing")
    .sum()
)


# ============================================================
# STATUS BANNER
# ============================================================

if data_mode == "Uploaded Dataset":

    st.info(
        f"📤 Live dataset mode — "
        f"{len(active_data):,} records loaded. "
        f"Forecast horizon: {forecast_days} days."
    )

else:

    st.warning(
        f"ℹ️ Demo mode — using the project dataset. "
        f"Forecast horizon: {forecast_days} days."
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    st.header(
        "📌 Dashboard Overview"
    )

    st.caption(
        "Executive view of demand, inventory, risk and forecasting intelligence."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "📦 Products",
            f"{total_products:,}"
        )

    with c2:

        st.metric(
            "🏭 Total Inventory",
            f"{total_inventory:,.0f}"
        )

    with c3:

        st.metric(
            f"📈 {forecast_days}-Day Forecast",
            f"{total_forecast:,.0f}"
        )

    with c4:

        st.metric(
            "🚨 Stockout Risks",
            f"{stockout_count:,}"
        )

    with c5:

        st.metric(
            "⬆️ Increasing Demand",
            f"{increasing_count:,}"
        )

    st.divider()

    category = (
        product_forecast[
            "Category"
        ].iloc[0]
        if not product_forecast.empty
        else "Unknown"
    )

    st.header(
        f"📦 {selected_product} — {category}"
    )

    current_inventory = (
        float(
            product_inventory[
                "Inventory Level"
            ].iloc[0]
        )
        if not product_inventory.empty
        else 0
    )

    forecast_period_demand = (
        float(
            product_forecast[
                "Predicted Demand"
            ].sum()
        )
    )

    average_daily_demand = (
        float(
            product_forecast[
                "Predicted Demand"
            ].mean()
        )
        if not product_forecast.empty
        else 0
    )

    reorder_quantity = (
        float(
            product_inventory[
                "Recommended Reorder Quantity"
            ].iloc[0]
        )
        if not product_inventory.empty
        else 0
    )

    coverage_days = (
        float(
            product_inventory[
                "Inventory Coverage Days"
            ].iloc[0]
        )
        if not product_inventory.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Current Inventory",
            f"{current_inventory:,.0f}"
        )

    with c2:

        st.metric(
            f"{forecast_days}-Day Demand",
            f"{forecast_period_demand:,.1f}"
        )

    with c3:

        st.metric(
            "Average Daily Demand",
            f"{average_daily_demand:,.1f}"
        )

    with c4:

        st.metric(
            "Recommended Reorder",
            f"{reorder_quantity:,.0f}"
        )

    st.subheader(
        "🚦 Inventory Status"
    )

    if not product_inventory.empty:

        status = product_inventory[
            "Inventory Status"
        ].iloc[0]

        action = product_inventory[
            "Recommended Action"
        ].iloc[0]

        if status == "STOCKOUT RISK":

            st.error(
                f"🔴 {status} — {action}"
            )

        elif status == "LOW STOCK":

            st.warning(
                f"🟠 {status} — {action}"
            )

        elif status == "OVERSTOCK":

            st.info(
                f"🔵 {status} — {action}"
            )

        else:

            st.success(
                f"🟢 {status} — {action}"
            )

    st.subheader(
        "📦 Inventory Coverage"
    )

    if coverage_days < 2:

        st.error(
            f"⚠️ Inventory covers approximately "
            f"{coverage_days:.1f} days of demand."
        )

    elif coverage_days < 7:

        st.warning(
            f"Inventory covers approximately "
            f"{coverage_days:.1f} days of demand."
        )

    else:

        st.success(
            f"Inventory covers approximately "
            f"{coverage_days:.1f} days of demand."
        )

    st.subheader(
        "📈 Demand Trend Intelligence"
    )

    if not product_trend.empty:

        row = product_trend.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Previous Average",
                f"{row['Previous Average Demand']:.2f}"
            )

        with c2:

            st.metric(
                "Recent Average",
                f"{row['Recent Average Demand']:.2f}"
            )

        with c3:

            st.metric(
                "Demand Change",
                f"{row['Demand Change %']:.2f}%"
            )

        with c4:

            st.metric(
                "Trend",
                row["Demand Trend"]
            )

    st.subheader(
        f"🔮 {forecast_days}-Day Demand Forecast"
    )

    if not product_forecast.empty:

        st.line_chart(
            product_forecast
            .set_index("Date")[
                "Predicted Demand"
            ]
        )


# ============================================================
# EDA PAGE
# ============================================================

elif page == "📊 Exploratory Data Analysis":

    st.header(
        "📊 Exploratory Data Analysis"
    )

    st.caption(
        "Analysis of data quality, demand distribution, "
        "historical trends and business variables."
    )

    date_min = active_data["Date"].min()
    date_max = active_data["Date"].max()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Total Records",
            f"{len(active_data):,}"
        )

    with c2:

        st.metric(
            "Stores",
            f"{active_data['Store ID'].nunique():,}"
        )

    with c3:

        st.metric(
            "Products",
            f"{active_data['Product ID'].nunique():,}"
        )

    with c4:

        st.metric(
            "Categories",
            f"{active_data['Category'].nunique():,}"
        )

    with c5:

        st.metric(
            "Total Demand",
            f"{active_data['Demand'].sum():,.0f}"
        )

    st.write(
        f"**Data period:** "
        f"{date_min:%Y-%m-%d} → "
        f"{date_max:%Y-%m-%d}"
    )

    st.divider()

    st.subheader(
        "🧹 Data Quality Analysis"
    )

    quality_rows = []

    for col in REQUIRED_COLUMNS:

        if col in active_data.columns:

            quality_rows.append(
                {
                    "Column": col,
                    "Missing Values": int(
                        active_data[col]
                        .isna()
                        .sum()
                    ),
                    "Unique Values": int(
                        active_data[col]
                        .nunique()
                    ),
                    "Data Type": str(
                        active_data[col].dtype
                    )
                }
            )

    quality_df = pd.DataFrame(
        quality_rows
    )

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    total_missing = int(
        active_data.isna()
        .sum()
        .sum()
    )

    if total_missing == 0:

        st.success(
            "✅ No missing values remain after data cleaning."
        )

    else:

        st.warning(
            f"⚠️ {total_missing:,} missing values remain."
        )

    st.subheader(
        "🧽 Data Cleaning Report"
    )

    if cleaning_report:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Invalid Dates Removed",
                f"{cleaning_report.get('Invalid Date Rows Removed', 0):,}"
            )

        with c2:

            st.metric(
                "Numeric Values Filled",
                f"{cleaning_report.get('Numeric Missing Values Filled', 0):,}"
            )

        with c3:

            st.metric(
                "Rows After Cleaning",
                f"{cleaning_report.get('Rows After Cleaning', len(active_data)):,}"
            )

    st.divider()

    st.subheader(
        "📈 Demand Statistics"
    )

    st.dataframe(
        active_data[
            "Demand"
        ]
        .describe()
        .to_frame("Demand")
        .round(2),
        use_container_width=True
    )

    st.subheader(
        "📉 Historical Demand Trend"
    )

    historical = (
        active_data
        .groupby("Date")["Demand"]
        .sum()
        .sort_index()
    )

    st.line_chart(
        historical
    )

    st.subheader(
        "📦 Demand by Category"
    )

    category_demand = (
        active_data
        .groupby("Category")["Demand"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_demand
    )

    st.subheader(
        "🌍 Demand by Region"
    )

    region_demand = (
        active_data
        .groupby("Region")["Demand"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        region_demand
    )

    st.subheader(
        "🔗 Business Variable Correlation"
    )

    available_numeric = [
        col
        for col in [
            "Demand",
            "Units Sold",
            "Units Ordered",
            "Inventory Level",
            "Price",
            "Discount",
            "Competitor Pricing"
        ]
        if col in active_data.columns
    ]

    if len(available_numeric) >= 2:

        correlation = (
            active_data[
                available_numeric
            ]
            .corr()
            .round(2)
        )

        st.dataframe(
            correlation,
            use_container_width=True
        )

    st.subheader(
        "💡 EDA Summary"
    )

    highest_category = (
        category_demand.index[0]
        if not category_demand.empty
        else "N/A"
    )

    highest_region = (
        region_demand.index[0]
        if not region_demand.empty
        else "N/A"
    )

    st.info(
        f"""
**Key observations**

• Average demand: **{active_data['Demand'].mean():.2f}**

• Highest-demand category: **{highest_category}**

• Highest-demand region: **{highest_region}**

• Records analyzed: **{len(active_data):,}**

• Stores: **{active_data['Store ID'].nunique():,}**

• Products: **{active_data['Product ID'].nunique():,}**
"""
    )


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "📈 Demand Forecast":

    st.header(
        f"🔮 {forecast_days}-Day Demand Forecast"
    )

    st.caption(
        "Demand forecast generated using the trained AI model "
        "with fallback forecasting when necessary."
    )

    history = active_data[
        (
            active_data["Store ID"] ==
            selected_store
        )
        &
        (
            active_data["Product ID"] ==
            selected_product
        )
    ][
        [
            "Date",
            "Demand"
        ]
    ].copy()

    history = (
        history
        .sort_values("Date")
        .tail(30)
        .rename(
            columns={
                "Demand":
                "Historical Demand"
            }
        )
    )

    predicted = product_forecast[
        [
            "Date",
            "Predicted Demand"
        ]
    ].copy()

    comparison = pd.concat(
        [
            history.set_index("Date")[
                "Historical Demand"
            ],
            predicted.set_index("Date")[
                "Predicted Demand"
            ]
        ],
        axis=1
    )

    st.subheader(
        "📊 Historical vs Predicted Demand"
    )

    st.line_chart(
        comparison
    )

    st.subheader(
        "📋 Forecast Details"
    )

    st.dataframe(
        product_forecast[
            [
                "Date",
                "Store ID",
                "Product ID",
                "Category",
                "Region",
                "Predicted Demand"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Selected Product Forecast",
        product_forecast.to_csv(
            index=False
        ),
        "selected_product_forecast.csv",
        "text/csv"
    )


# ============================================================
# INVENTORY PAGE
# ============================================================

elif page == "📦 Inventory Intelligence":

    st.header(
        "📦 Inventory Intelligence"
    )

    st.caption(
        "Inventory decisions are based on forecast demand, "
        "safety stock, reorder level and inventory coverage."
    )

    if not product_inventory.empty:

        row = product_inventory.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Inventory",
                f"{row['Inventory Level']:,.0f}"
            )

        with c2:

            st.metric(
                "Forecast Demand",
                f"{row['Predicted Demand']:,.0f}"
            )

        with c3:

            st.metric(
                "Safety Stock",
                f"{row['Safety Stock']:,.0f}"
            )

        with c4:

            st.metric(
                "Reorder Quantity",
                f"{row['Recommended Reorder Quantity']:,.0f}"
            )

        st.subheader(
            "📋 Product Inventory Analysis"
        )

        st.dataframe(
            product_inventory[
                [
                    "Inventory Level",
                    "Predicted Demand",
                    "Average Daily Demand",
                    "Inventory Coverage Days",
                    "Safety Stock",
                    "Reorder Level",
                    "Required Inventory",
                    "Stock Difference",
                    "Inventory Status",
                    "Risk Level",
                    "Recommended Reorder Quantity",
                    "Recommended Action"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "🚦 Inventory Risk Overview"
    )

    risk_counts = (
        inventory[
            "Inventory Status"
        ]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )

    st.subheader(
        "🚨 Products Requiring Attention"
    )

    risky_inventory = inventory[
        inventory["Inventory Status"]
        .isin(
            [
                "STOCKOUT RISK",
                "LOW STOCK",
                "OVERSTOCK"
            ]
        )
    ].sort_values(
        "Recommended Reorder Quantity",
        ascending=False
    )

    if risky_inventory.empty:

        st.success(
            "✅ No significant inventory risks detected."
        )

    else:

        st.dataframe(
            risky_inventory[
                [
                    "Store ID",
                    "Product ID",
                    "Category",
                    "Inventory Level",
                    "Predicted Demand",
                    "Inventory Status",
                    "Risk Level",
                    "Recommended Reorder Quantity",
                    "Recommended Action"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DEMAND PATTERNS
# ============================================================

elif page == "🔎 Demand Patterns":

    st.header(
        "🔎 Demand Patterns & Seasonality"
    )

    st.caption(
        "Identify weekly, monthly, seasonal and product-level "
        "demand behavior."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "📅 Weekly Pattern"
        )

        if not weekly_pattern.empty:

            high = weekly_pattern.loc[
                weekly_pattern[
                    "Average Demand"
                ].idxmax()
            ]

            low = weekly_pattern.loc[
                weekly_pattern[
                    "Average Demand"
                ].idxmin()
            ]

            st.write(
                f"Highest-demand day: "
                f"**{high['Day']}**"
            )

            st.write(
                f"Lowest-demand day: "
                f"**{low['Day']}**"
            )

            st.bar_chart(
                weekly_pattern
                .set_index("Day")[
                    "Average Demand"
                ]
            )

    with c2:

        st.subheader(
            "📆 Monthly Pattern"
        )

        if not monthly_pattern.empty:

            high = monthly_pattern.loc[
                monthly_pattern[
                    "Average Demand"
                ].idxmax()
            ]

            low = monthly_pattern.loc[
                monthly_pattern[
                    "Average Demand"
                ].idxmin()
            ]

            st.write(
                f"Highest-demand month: "
                f"**{high['MonthName']}**"
            )

            st.write(
                f"Lowest-demand month: "
                f"**{low['MonthName']}**"
            )

            st.bar_chart(
                monthly_pattern
                .set_index("MonthName")[
                    "Average Demand"
                ]
            )

    st.subheader(
        "🌦 Seasonal Demand Pattern"
    )

    if not seasonal_pattern.empty:

        high = seasonal_pattern.loc[
            seasonal_pattern[
                "Average Demand"
            ].idxmax()
        ]

        low = seasonal_pattern.loc[
            seasonal_pattern[
                "Average Demand"
            ].idxmin()
        ]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Highest-Demand Season",
                str(
                    high["Seasonality"]
                )
            )

        with c2:

            st.metric(
                "Lowest-Demand Season",
                str(
                    low["Seasonality"]
                )
            )

        st.bar_chart(
            seasonal_pattern
            .set_index("Seasonality")[
                "Average Demand"
            ]
        )

    st.subheader(
        "📈 Product Demand Trend Summary"
    )

    trend_summary = (
        trend[
            "Demand Trend"
        ]
        .value_counts()
    )

    st.bar_chart(
        trend_summary
    )


# ============================================================
# AI BUSINESS INSIGHTS
# ============================================================

elif page == "🧠 AI Business Insights":

    st.header(
        "🧠 Actionable AI Business Insights"
    )

    st.caption(
        "Technical results translated into practical business decisions."
    )

    total_inventory_products = len(
        inventory
    )

    stockout_percentage = (
        stockout_count /
        total_inventory_products *
        100
        if total_inventory_products > 0
        else 0
    )

    overstock_percentage = (
        overstock_count /
        total_inventory_products *
        100
        if total_inventory_products > 0
        else 0
    )

    increasing_percentage = (
        increasing_count /
        len(trend) *
        100
        if len(trend) > 0
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Stockout Risk %",
            f"{stockout_percentage:.1f}%"
        )

    with c2:

        st.metric(
            "Overstock %",
            f"{overstock_percentage:.1f}%"
        )

    with c3:

        st.metric(
            "Increasing Demand %",
            f"{increasing_percentage:.1f}%"
        )

    st.divider()

    st.subheader(
        "🚨 Priority Recommendations"
    )

    high_risk = inventory[
        inventory["Inventory Status"]
        == "STOCKOUT RISK"
    ].sort_values(
        "Recommended Reorder Quantity",
        ascending=False
    )

    overstock = inventory[
        inventory["Inventory Status"]
        == "OVERSTOCK"
    ].sort_values(
        "Stock Difference",
        ascending=False
    )

    increasing_products = trend[
        trend["Demand Trend"]
        == "Increasing"
    ].sort_values(
        "Demand Change %",
        ascending=False
    )

    if not high_risk.empty:

        st.error(
            f"🚨 **{len(high_risk)} product/store combinations "
            f"are at stockout risk.** "
            f"Prioritize replenishment."
        )

        st.dataframe(
            high_risk[
                [
                    "Store ID",
                    "Product ID",
                    "Inventory Level",
                    "Predicted Demand",
                    "Recommended Reorder Quantity"
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "✅ No immediate stockout-risk products detected."
        )

    if not overstock.empty:

        st.info(
            f"📦 **{len(overstock)} product/store combinations "
            f"show overstock conditions.** "
            f"Review purchasing and redistribution."
        )

    if not increasing_products.empty:

        st.warning(
            f"📈 **{len(increasing_products)} product/store combinations "
            f"show increasing demand.** "
            f"Consider increasing replenishment planning."
        )

    st.divider()

    st.subheader(
        f"🎯 Recommendation for {selected_product}"
    )

    if not product_inventory.empty:

        row = product_inventory.iloc[0]

        status = row[
            "Inventory Status"
        ]

        if status == "STOCKOUT RISK":

            st.error(
                "🔴 Action: Reorder immediately."
            )

        elif status == "LOW STOCK":

            st.warning(
                "🟠 Action: Increase inventory toward "
                "the recommended level."
            )

        elif status == "OVERSTOCK":

            st.info(
                "🔵 Action: Reduce incoming orders "
                "and consider redistribution."
            )

        else:

            st.success(
                "🟢 Action: Maintain the current inventory strategy."
            )

        st.write(
            f"**Recommended reorder quantity:** "
            f"{row['Recommended Reorder Quantity']:,.0f}"
        )

        st.write(
            f"**Safety stock:** "
            f"{row['Safety Stock']:,.0f}"
        )

        st.write(
            f"**Inventory coverage:** "
            f"{row['Inventory Coverage Days']:.1f} days"
        )

        st.write(
            f"**Recommendation:** "
            f"{row['Recommended Action']}"
        )


# ============================================================
# DATA & WORKFLOW
# ============================================================

elif page == "📚 Data & Workflow":

    st.header(
        "📚 Data Requirements & Application Workflow"
    )

    st.caption(
        "Documentation of the dataset, features, processing pipeline "
        "and application workflow."
    )

    st.subheader(
        "📋 Dataset Requirements"
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Required Feature": REQUIRED_COLUMNS,

                "Description": [
                    DATA_DICTIONARY[col][0]
                    for col in REQUIRED_COLUMNS
                ],

                "Application Usage": [
                    DATA_DICTIONARY[col][1]
                    for col in REQUIRED_COLUMNS
                ]
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Product Name is not present in the current dataset. "
        "Product ID is used as the product identifier and Category "
        "provides product classification."
    )

    

    st.divider()

    st.subheader(
        "⚙️ Feature Engineering"
    )

    feature_info = pd.DataFrame(
        {
            "Engineered Feature": [

                "DayOfWeek",
                "DayOfMonth",
                "WeekOfYear",
                "Month",
                "Quarter",
                "Year",
                "IsWeekend",
                "IsHoliday",

                "Demand Lag 1",
                "Demand Lag 7",
                "Demand Lag 14",
                "Demand Lag 30",

                "Demand Rolling Mean 7",
                "Demand Rolling Mean 14",
                "Demand Rolling Mean 30",

                "Demand Rolling Std 7",
                "Demand Rolling Std 14",

                "Demand Rolling Min 7",
                "Demand Rolling Max 7",

                "Historical Demand",

                "Previous Price",
                "Price Change",
                "Price_Change",
                "Price Change %",

                "Promotion Indicator",
                "Product Category",
                "Store Region",

                "Discount Feature",
                "Competitor Price Difference"
            ]
        }
    )

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "✅ Feature engineering is active, including "
        "Quarter, IsWeekend and Price_Change required by the AI model."
    )

    st.subheader(
        "🔍 Engineered Feature Preview"
    )

    preview_columns = [

        "Date",
        "Store ID",
        "Product ID",
        "Demand",

        "DayOfWeek",
        "WeekOfYear",
        "Month",
        "Quarter",
        "IsWeekend",
        "IsHoliday",

        "Demand Lag 1",
        "Demand Lag 7",
        "Demand Lag 14",
        "Demand Lag 30",

        "Demand Rolling Mean 7",
        "Demand Rolling Mean 14",
        "Demand Rolling Mean 30",

        "Demand Rolling Std 7",

        "Historical Demand",

        "Price Change",
        "Price_Change",
        "Price Change %",

        "Promotion Indicator",

        "Product Category"
    ]

    available_preview_columns = [
        col
        for col in preview_columns
        if col in engineered_data.columns
    ]

    st.dataframe(
        engineered_data[
            available_preview_columns
        ].tail(20),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "📊 Current Dataset Status"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Records",
            f"{len(active_data):,}"
        )

    with c2:

        st.metric(
            "Original Features",
            f"{len(REQUIRED_COLUMNS):,}"
        )

    with c3:

        st.metric(
            "Stores",
            f"{active_data['Store ID'].nunique():,}"
        )

    with c4:

        st.metric(
            "Products",
            f"{active_data['Product ID'].nunique():,}"
        )

    st.metric(
        "Engineered Features",
        f"{len(engineered_data.columns) - len(active_data.columns):,}"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Model Comparison & Performance"
    )

    st.caption(
        "Evaluation results from the project's model-comparison stage."
    )

    if ai_model is not None:

        st.success(
            "✅ Random Forest model loaded successfully from "
            "`models/random_forest.pkl`."
        )

    else:

        st.warning(
            "⚠️ Random Forest model was not found."
        )

    if model_comparison.empty:

        st.warning(
            "No model comparison file found at "
            "`data/processed/model_comparison.csv`."
        )

    else:

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

        if "Model" in model_comparison.columns:

            metrics = [
                col
                for col in [
                    "MAE",
                    "RMSE",
                    "MAPE"
                ]
                if col in model_comparison.columns
            ]

            if metrics:

                st.subheader(
                    "📊 Model Evaluation"
                )

                st.bar_chart(
                    model_comparison
                    .set_index("Model")[
                        metrics
                    ]
                )

        if "MAE" in model_comparison.columns:

            best_row = (
                model_comparison
                .sort_values("MAE")
                .iloc[0]
            )

            st.success(
                f"🏆 Best model according to MAE: "
                f"**{best_row['Model']}**"
            )


# ============================================================
# SYSTEM VALIDATION
# ============================================================

elif page == "⚙️ System Validation":

    st.header(
        "⚙️ System Validation"
    )

    st.caption(
        "Checks whether the application workflow and required "
        "project components are available."
    )

    checks = []

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Dataset",
            "Status": "PASS",
            "Details": (
                f"{len(active_data):,} records loaded"
            )
        }
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns, extra_columns = (
        validate_dataset(
            active_data
        )
    )

    checks.append(
        {
            "Component": "Required Columns",
            "Status": (
                "PASS"
                if not missing_columns
                else "FAIL"
            ),
            "Details": (
                "All required columns available"
                if not missing_columns
                else
                ", ".join(
                    missing_columns
                )
            )
        }
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Date Validation",
            "Status": (
                "PASS"
                if active_data["Date"]
                .notna()
                .all()
                else "WARNING"
            ),
            "Details": (
                "All active dates are valid"
                if active_data["Date"]
                .notna()
                .all()
                else
                "Invalid dates remain"
            )
        }
    )

    # --------------------------------------------------------
    # DEMAND
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Demand Data",
            "Status": (
                "PASS"
                if active_data["Demand"]
                .notna()
                .all()
                else "WARNING"
            ),
            "Details": "Demand target available"
        }
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    required_engineered_features = [

        "Demand Lag 1",
        "Demand Lag 7",

        "Demand Rolling Mean 7",
        "Demand Rolling Mean 14",

        "DayOfWeek",
        "WeekOfYear",
        "Month",
        "Quarter",
        "IsWeekend",

        "IsHoliday",

        "Promotion Indicator",

        "Price Change",
        "Price_Change",

        "Historical Demand",

        "Product Category"
    ]

    missing_engineered_features = [

        col
        for col in required_engineered_features
        if col not in engineered_data.columns
    ]

    checks.append(
        {
            "Component": "Feature Engineering",
            "Status": (
                "PASS"
                if not missing_engineered_features
                else "FAIL"
            ),
            "Details": (
                "All model-compatible features generated"
                if not missing_engineered_features
                else
                ", ".join(
                    missing_engineered_features
                )
            )
        }
    )

    # --------------------------------------------------------
    # AI MODEL
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Saved AI Model",
            "Status": (
                "PASS"
                if ai_model is not None
                else "WARNING"
            ),
            "Details": (
                "Random Forest model loaded successfully"
                if ai_model is not None
                else
                "Model unavailable; fallback forecasting active"
            )
        }
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Forecast Engine",
            "Status": (
                "PASS"
                if not forecast.empty
                else "FAIL"
            ),
            "Details": (
                f"{len(forecast):,} forecast records generated"
            )
        }
    )

    # --------------------------------------------------------
    # INVENTORY
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Inventory Analysis",
            "Status": (
                "PASS"
                if not inventory.empty
                else "FAIL"
            ),
            "Details": (
                f"{len(inventory):,} product/store analyses generated"
            )
        }
    )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Actionable Insights",
            "Status": "PASS",
            "Details": (
                "Stockout, overstock and demand recommendations available"
            )
        }
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    checks.append(
        {
            "Component": "Model Comparison",
            "Status": (
                "PASS"
                if not model_comparison.empty
                else "WARNING"
            ),
            "Details": (
                "Model comparison file loaded"
                if not model_comparison.empty
                else
                "Comparison file not found"
            )
        }
    )

    validation_df = pd.DataFrame(
        checks
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )

    passed = int(
        (
            validation_df["Status"]
            == "PASS"
        ).sum()
    )

    total_checks = len(
        validation_df
    )

    st.metric(
        "System Checks Passed",
        f"{passed}/{total_checks}"
    )

    if passed == total_checks:

        st.success(
            "✅ All major application workflow components are operational."
        )

    else:

        st.warning(
            "⚠️ Some project components require attention."
        )


# ============================================================
# DOWNLOAD REPORTS
# ============================================================

st.divider()

st.header(
    "📥 Download Reports"
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.download_button(
        "📈 Forecast",
        forecast.to_csv(
            index=False
        ),
        "future_demand_forecast.csv",
        "text/csv"
    )

with c2:

    st.download_button(
        "📦 Inventory",
        inventory.to_csv(
            index=False
        ),
        "inventory_analysis.csv",
        "text/csv"
    )

with c3:

    st.download_button(
        "📊 Trends",
        trend.to_csv(
            index=False
        ),
        "demand_trend_analysis.csv",
        "text/csv"
    )

with c4:

    st.download_button(
        "🤖 Model Results",
        model_comparison.to_csv(
            index=False
        ),
        "model_comparison.csv",
        "text/csv"
    )

with c5:

    st.download_button(
        "🧹 Clean Dataset",
        active_data.to_csv(
            index=False
        ),
        "cleaned_dataset.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Demand Intelligence • Demand Forecasting • "
    "Inventory Optimization • Actionable Business Insights"
)
