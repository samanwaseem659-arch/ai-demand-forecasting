import os
import joblib
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "models/random_forest.pkl"
DATA_PATH = "data/processed/features.csv"

OUTPUT_PATH = (
    "data/processed/future_demand_forecast.csv"
)

FORECAST_DAYS = 7


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Saved model loaded successfully!")


# ============================================================
# 3. LOAD FEATURE DATA
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
).copy()


# ============================================================
# 4. SORT DATA
# ============================================================

df = (
    df.sort_values(
        ["Store ID", "Product ID", "Date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 5. MODEL FEATURES
# ============================================================

features = [

    # Business features
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",

    # Calendar features
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "Quarter",
    "IsWeekend",

    # Lag features
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",

    # Rolling features
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    # Price change
    "Price_Change",

    # Categorical features
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]


# ============================================================
# 6. CHECK FEATURES
# ============================================================

missing_features = [
    col
    for col in features
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# 7. LATEST DATE
# ============================================================

latest_date = df["Date"].max()

print(
    f"\nLatest historical date: "
    f"{latest_date.date()}"
)

print(
    f"Forecast horizon: "
    f"{FORECAST_DAYS} days"
)


# ============================================================
# 8. FORECAST FUNCTION
# ============================================================

def forecast_product(
    history,
    forecast_days=7
):

    history = (
        history
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Need enough historical demand for 30-day rolling mean
    # --------------------------------------------------------

    if len(history) < 30:

        return []

    # --------------------------------------------------------
    # Store and product information
    # --------------------------------------------------------

    store_id = history["Store ID"].iloc[-1]

    product_id = history["Product ID"].iloc[-1]

    category = history["Category"].iloc[-1]

    region = history["Region"].iloc[-1]

    weather = history["Weather Condition"].iloc[-1]

    seasonality = history["Seasonality"].iloc[-1]

    # --------------------------------------------------------
    # Keep historical demand
    # --------------------------------------------------------

    demand_history = (
        history["Demand"]
        .astype(float)
        .tolist()
    )

    # --------------------------------------------------------
    # Last known business values
    #
    # For future dates, these values are carried forward
    # because future business conditions are not available
    # in the current dataset.
    # --------------------------------------------------------

    last_row = history.iloc[-1]

    inventory_level = last_row["Inventory Level"]

    units_ordered = last_row["Units Ordered"]

    price = last_row["Price"]

    discount = last_row["Discount"]

    promotion = last_row["Promotion"]

    competitor_pricing = last_row["Competitor Pricing"]

    epidemic = last_row["Epidemic"]

    # --------------------------------------------------------
    # Future forecast results
    # --------------------------------------------------------

    forecasts = []

    current_date = history["Date"].max()

    # ========================================================
    # RECURSIVE FORECASTING
    # ========================================================

    for step in range(1, forecast_days + 1):

        future_date = (
            current_date
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        year = future_date.year

        month = future_date.month

        day = future_date.day

        day_of_week = future_date.dayofweek

        week_of_year = (
            future_date.isocalendar().week
        )

        quarter = future_date.quarter

        is_weekend = (
            1
            if day_of_week >= 5
            else 0
        )

        # ----------------------------------------------------
        # Demand lag features
        # ----------------------------------------------------

        lag_1 = demand_history[-1]

        if len(demand_history) >= 7:

            lag_7 = demand_history[-7]

        else:

            lag_7 = demand_history[0]

        if len(demand_history) >= 14:

            lag_14 = demand_history[-14]

        else:

            lag_14 = demand_history[0]

        # ----------------------------------------------------
        # Rolling demand features
        # ----------------------------------------------------

        rolling_mean_7 = (
            sum(demand_history[-7:])
            / min(7, len(demand_history))
        )

        rolling_mean_14 = (
            sum(demand_history[-14:])
            / min(14, len(demand_history))
        )

        rolling_mean_30 = (
            sum(demand_history[-30:])
            / min(30, len(demand_history))
        )

        # ----------------------------------------------------
        # Future price change
        #
        # Since no future price is available, we assume
        # the latest price remains unchanged.
        # ----------------------------------------------------

        price_change = 0.0

        # ----------------------------------------------------
        # Create prediction row
        # ----------------------------------------------------

        future_row = pd.DataFrame(
            [
                {

                    # Business features
                    "Inventory Level":
                        inventory_level,

                    "Units Ordered":
                        units_ordered,

                    "Price":
                        price,

                    "Discount":
                        discount,

                    "Promotion":
                        promotion,

                    "Competitor Pricing":
                        competitor_pricing,

                    "Epidemic":
                        epidemic,

                    # Calendar features
                    "Year":
                        year,

                    "Month":
                        month,

                    "Day":
                        day,

                    "DayOfWeek":
                        day_of_week,

                    "WeekOfYear":
                        int(week_of_year),

                    "Quarter":
                        quarter,

                    "IsWeekend":
                        is_weekend,

                    # Lag features
                    "Demand_Lag_1":
                        lag_1,

                    "Demand_Lag_7":
                        lag_7,

                    "Demand_Lag_14":
                        lag_14,

                    # Rolling features
                    "Demand_Rolling_Mean_7":
                        rolling_mean_7,

                    "Demand_Rolling_Mean_14":
                        rolling_mean_14,

                    "Demand_Rolling_Mean_30":
                        rolling_mean_30,

                    # Price change
                    "Price_Change":
                        price_change,

                    # Categorical features
                    "Store ID":
                        store_id,

                    "Product ID":
                        product_id,

                    "Category":
                        category,

                    "Region":
                        region,

                    "Weather Condition":
                        weather,

                    "Seasonality":
                        seasonality
                }
            ]
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        predicted_demand = model.predict(
            future_row[features]
        )[0]

        # Demand cannot be negative
        predicted_demand = max(
            0,
            float(predicted_demand)
        )

        # Round prediction
        predicted_demand = round(
            predicted_demand,
            2
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        forecasts.append(
            {

                "Date":
                    future_date,

                "Store ID":
                    store_id,

                "Product ID":
                    product_id,

                "Category":
                    category,

                "Region":
                    region,

                "Predicted Demand":
                    predicted_demand,

                "Inventory Level":
                    inventory_level,

                "Units Ordered":
                    units_ordered,

                "Price":
                    price,

                "Promotion":
                    promotion

            }
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Add prediction to demand history so the next
        # forecast day can use it as a lag.
        # ----------------------------------------------------

        demand_history.append(
            predicted_demand
        )

        current_date = future_date

    return forecasts


# ============================================================
# 9. FORECAST ALL STORE + PRODUCT COMBINATIONS
# ============================================================

print(
    "\nGenerating future demand forecasts..."
)

all_forecasts = []

groups = df.groupby(
    ["Store ID", "Product ID"]
)

total_groups = len(groups)

print(
    f"Store/Product combinations: "
    f"{total_groups}"
)


for index, (
    (store_id, product_id),
    group
) in enumerate(groups, start=1):

    results = forecast_product(
        group,
        forecast_days=FORECAST_DAYS
    )

    all_forecasts.extend(
        results
    )

    # Progress message
    if index % 100 == 0:

        print(
            f"Processed "
            f"{index}/{total_groups} "
            f"store/product combinations..."
        )


# ============================================================
# 10. CREATE FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    all_forecasts
)


if forecast_df.empty:

    raise ValueError(
        "No forecasts were generated. "
        "Check whether each Store/Product group "
        "has at least 30 historical records."
    )


# ============================================================
# 11. SORT RESULTS
# ============================================================

forecast_df = (
    forecast_df
    .sort_values(
        [
            "Date",
            "Store ID",
            "Product ID"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. SAVE FORECAST
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print()
print(
    "============================================"
)

print(
    "FUTURE DEMAND FORECAST COMPLETED"
)

print(
    "============================================"
)

print(
    f"Forecast dates: "
    f"{forecast_df['Date'].min().date()} "
    f"to "
    f"{forecast_df['Date'].max().date()}"
)

print(
    f"Forecast records: "
    f"{len(forecast_df):,}"
)

print(
    f"Store/Product combinations: "
    f"{forecast_df[['Store ID', 'Product ID']].drop_duplicates().shape[0]:,}"
)

print()
print(
    "Sample forecasts:"
)

print(
    forecast_df.head(20).to_string(
        index=False
    )
)

print()
print(
    "Forecast saved successfully!"
)

print(
    f"Location: {OUTPUT_PATH}"
)