import os
import pandas as pd


def create_features():

    # ==================================================
    # 1. LOAD CLEANED DATASET
    # ==================================================

    input_path = "data/processed/sales_cleaned.csv"

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["Date"]
    ).copy()

    # ==================================================
    # 2. SORT DATA
    # ==================================================

    df = (
        df.sort_values(
            ["Store ID", "Product ID", "Date"]
        )
        .reset_index(drop=True)
    )

    # ==================================================
    # 3. CALENDAR FEATURES
    # ==================================================

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Day"] = df["Date"].dt.day

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["WeekOfYear"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["Quarter"] = (
        df["Date"]
        .dt.quarter
    )

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # ==================================================
    # 4. GROUP BY STORE + PRODUCT
    # ==================================================

    group = df.groupby(
        ["Store ID", "Product ID"]
    )

    # ==================================================
    # 5. LAG FEATURES
    # ==================================================

    df["Demand_Lag_1"] = (
        group["Demand"]
        .shift(1)
    )

    df["Demand_Lag_7"] = (
        group["Demand"]
        .shift(7)
    )

    df["Demand_Lag_14"] = (
        group["Demand"]
        .shift(14)
    )

    # ==================================================
    # 6. ROLLING DEMAND FEATURES
    # ==================================================

    df["Demand_Rolling_Mean_7"] = (
        group["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=7
            )
            .mean()
        )
    )

    df["Demand_Rolling_Mean_14"] = (
        group["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=14,
                min_periods=14
            )
            .mean()
        )
    )

    df["Demand_Rolling_Mean_30"] = (
        group["Demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=30,
                min_periods=30
            )
            .mean()
        )
    )

    # ==================================================
    # 7. PRICE CHANGE
    # ==================================================

    df["Price_Change"] = (
        group["Price"]
        .pct_change()
    )

    # Replace infinite values
    df["Price_Change"] = (
        df["Price_Change"]
        .replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
    )

    # ==================================================
    # 8. SAVE FEATURE DATASET
    # ==================================================

    output_path = (
        "data/processed/features.csv"
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    # ==================================================
    # 9. DISPLAY INFORMATION
    # ==================================================

    print(
        "Feature engineering completed!"
    )

    print(
        "Feature dataset shape:",
        df.shape
    )

    print("\nCreated features:")

    print(
        [
            "Year",
            "Month",
            "Day",
            "DayOfWeek",
            "WeekOfYear",
            "Quarter",
            "IsWeekend",
            "Demand_Lag_1",
            "Demand_Lag_7",
            "Demand_Lag_14",
            "Demand_Rolling_Mean_7",
            "Demand_Rolling_Mean_14",
            "Demand_Rolling_Mean_30",
            "Price_Change"
        ]
    )


if __name__ == "__main__":
    create_features()