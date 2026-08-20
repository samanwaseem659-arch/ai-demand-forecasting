import joblib
import pandas as pd
import numpy as np
import os


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model_path = "models/random_forest.pkl"

model = joblib.load(model_path)

print("Saved model loaded successfully!")


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

df = pd.read_csv(
    "data/processed/features.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    ["Store ID", "Product ID", "Date"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE REQUIRED FEATURES
# ============================================================

# ------------------------------------------------------------
# TIME FEATURES
# ------------------------------------------------------------

if "Year" not in df.columns:
    df["Year"] = df["Date"].dt.year

if "Month" not in df.columns:
    df["Month"] = df["Date"].dt.month

if "Day" not in df.columns:
    df["Day"] = df["Date"].dt.day

if "DayOfWeek" not in df.columns:
    df["DayOfWeek"] = df["Date"].dt.dayofweek

if "WeekOfYear" not in df.columns:
    df["WeekOfYear"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

if "Quarter" not in df.columns:
    df["Quarter"] = (
        df["Date"]
        .dt.quarter
    )

if "IsWeekend" not in df.columns:
    df["IsWeekend"] = (
        df["Date"]
        .dt.dayofweek >= 5
    ).astype(int)


# ------------------------------------------------------------
# PRICE CHANGE
# ------------------------------------------------------------

if "Price_Change" not in df.columns:

    df["Price_Change"] = (
        df.groupby(
            ["Store ID", "Product ID"]
        )["Price"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# 4. SORT AGAIN
# ============================================================

df = df.sort_values(
    ["Store ID", "Product ID", "Date"]
).reset_index(drop=True)


# ============================================================
# 5. MODEL FEATURES
# ============================================================

features = [
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",

    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "Quarter",
    "IsWeekend",

    "Price_Change",

    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",

    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]


# ============================================================
# 6. CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise ValueError(
        "The feature dataset does not contain all "
        "features required by the trained model."
    )


# ============================================================
# 7. GET LATEST RECORD FOR EACH STORE + PRODUCT
# ============================================================

latest_date = df["Date"].max()

print(
    "\nLatest available date:",
    latest_date
)


latest_records = (
    df
    .sort_values("Date")
    .groupby(
        ["Store ID", "Product ID"]
    )
    .tail(1)
    .copy()
)


print(
    "Store/Product combinations:",
    len(latest_records)
)


# ============================================================
# 8. FORECAST SETTINGS
# ============================================================

forecast_days = 7

forecast_results = []


# ============================================================
# 9. FORECAST EACH STORE / PRODUCT
# ============================================================

for _, row in latest_records.iterrows():

    store_id = row["Store ID"]

    product_id = row["Product ID"]


    # --------------------------------------------------------
    # PRODUCT HISTORY
    # --------------------------------------------------------

    history = df[
        (df["Store ID"] == store_id)
        &
        (df["Product ID"] == product_id)
    ].sort_values("Date").copy()


    demand_history = (
        history["Demand"]
        .tolist()
    )


    # --------------------------------------------------------
    # PRICE HISTORY
    # --------------------------------------------------------

    price_history = (
        history["Price"]
        .tolist()
    )


    previous_date = row["Date"]


    # ========================================================
    # FUTURE DAYS
    # ========================================================

    for day in range(
        1,
        forecast_days + 1
    ):

        future_date = (
            previous_date
            + pd.Timedelta(days=1)
        )


        # ----------------------------------------------------
        # DEMAND LAGS
        # ----------------------------------------------------

        lag_1 = (
            demand_history[-1]
            if len(demand_history) >= 1
            else 0
        )


        lag_7 = (
            demand_history[-7]
            if len(demand_history) >= 7
            else np.mean(demand_history)
        )


        lag_14 = (
            demand_history[-14]
            if len(demand_history) >= 14
            else np.mean(demand_history)
        )


        # ----------------------------------------------------
        # ROLLING DEMAND
        # ----------------------------------------------------

        rolling_7 = np.mean(
            demand_history[-7:]
        )


        rolling_14 = np.mean(
            demand_history[-14:]
        )


        rolling_30 = np.mean(
            demand_history[-30:]
        )


        # ----------------------------------------------------
        # TIME FEATURES
        # ----------------------------------------------------

        future_year = (
            future_date.year
        )

        future_month = (
            future_date.month
        )

        future_day = (
            future_date.day
        )

        future_day_of_week = (
            future_date.dayofweek
        )

        future_week = int(
            future_date.isocalendar().week
        )

        future_quarter = (
            future_date.quarter
        )

        future_is_weekend = int(
            future_day_of_week >= 5
        )


        # ----------------------------------------------------
        # PRICE CHANGE
        # ----------------------------------------------------

        current_price = float(
            row["Price"]
        )


        if len(price_history) >= 1:

            previous_price = float(
                price_history[-1]
            )

        else:

            previous_price = (
                current_price
            )


        if previous_price != 0:

            price_change = (
                current_price -
                previous_price
            ) / previous_price

        else:

            price_change = 0


        # ----------------------------------------------------
        # CREATE FUTURE ROW
        # ----------------------------------------------------

        future_row = pd.DataFrame(
            [
                {

                    "Inventory Level":
                        row["Inventory Level"],

                    "Units Ordered":
                        row["Units Ordered"],

                    "Price":
                        current_price,

                    "Discount":
                        row["Discount"],

                    "Promotion":
                        row["Promotion"],

                    "Competitor Pricing":
                        row["Competitor Pricing"],

                    "Epidemic":
                        row["Epidemic"],


                    # TIME
                    "Year":
                        future_year,

                    "Month":
                        future_month,

                    "Day":
                        future_day,

                    "DayOfWeek":
                        future_day_of_week,

                    "WeekOfYear":
                        future_week,

                    "Quarter":
                        future_quarter,

                    "IsWeekend":
                        future_is_weekend,


                    # PRICE
                    "Price_Change":
                        price_change,


                    # DEMAND LAGS
                    "Demand_Lag_1":
                        lag_1,

                    "Demand_Lag_7":
                        lag_7,

                    "Demand_Lag_14":
                        lag_14,


                    # ROLLING DEMAND
                    "Demand_Rolling_Mean_7":
                        rolling_7,

                    "Demand_Rolling_Mean_14":
                        rolling_14,

                    "Demand_Rolling_Mean_30":
                        rolling_30,


                    # CATEGORICAL
                    "Store ID":
                        row["Store ID"],

                    "Product ID":
                        row["Product ID"],

                    "Category":
                        row["Category"],

                    "Region":
                        row["Region"],

                    "Weather Condition":
                        row["Weather Condition"],

                    "Seasonality":
                        row["Seasonality"]
                }
            ]
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            future_row[features]
        )[0]


        prediction = max(
            0,
            float(prediction)
        )


        # ----------------------------------------------------
        # SAVE FORECAST RESULT
        # ----------------------------------------------------

        forecast_results.append(
            {
                "Date":
                    future_date,

                "Store ID":
                    store_id,

                "Product ID":
                    product_id,

                "Category":
                    row["Category"],

                "Region":
                    row["Region"],

                "Predicted Demand":
                    round(
                        prediction,
                        2
                    )
            }
        )


        # ----------------------------------------------------
        # UPDATE HISTORY
        # ----------------------------------------------------
        # Important:
        # Tomorrow's prediction can use today's
        # predicted demand as lag-1.

        demand_history.append(
            prediction
        )

        price_history.append(
            current_price
        )

        previous_date = (
            future_date
        )


# ============================================================
# 10. CREATE FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_results
)


# ============================================================
# 11. SAVE FORECAST
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


output_path = (
    "data/processed/"
    "future_demand_forecast.csv"
)


forecast_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FUTURE DEMAND FORECAST")
print("=" * 60)

print(
    forecast_df
    .head(30)
    .to_string(index=False)
)


print("\n")
print("=" * 60)
print("FORECAST COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Forecast horizon:",
    forecast_days,
    "days"
)

print(
    "Total forecast records:",
    len(forecast_df)
)

print(
    "Output file:",
    output_path
)