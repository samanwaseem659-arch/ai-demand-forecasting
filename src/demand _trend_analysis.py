import pandas as pd


# =========================================================
# 1. LOAD CLEANED DATA
# =========================================================

df = pd.read_csv(
    "data/processed/sales_cleaned.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Store ID", "Product ID", "Date"]
).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Records:", len(df))


# =========================================================
# 2. CALCULATE RECENT AND PREVIOUS DEMAND
# =========================================================

# Last 30 days available in the dataset
latest_date = df["Date"].max()

recent_start = latest_date - pd.Timedelta(days=29)
previous_start = latest_date - pd.Timedelta(days=59)
previous_end = latest_date - pd.Timedelta(days=30)


recent = df[
    (df["Date"] >= recent_start)
    &
    (df["Date"] <= latest_date)
]


previous = df[
    (df["Date"] >= previous_start)
    &
    (df["Date"] <= previous_end)
]


# =========================================================
# 3. CALCULATE AVERAGE DEMAND
# =========================================================

recent_demand = (
    recent
    .groupby(["Store ID", "Product ID"])["Demand"]
    .mean()
    .reset_index(name="Recent Average Demand")
)


previous_demand = (
    previous
    .groupby(["Store ID", "Product ID"])["Demand"]
    .mean()
    .reset_index(name="Previous Average Demand")
)


# =========================================================
# 4. COMBINE RESULTS
# =========================================================

trend = recent_demand.merge(
    previous_demand,
    on=["Store ID", "Product ID"],
    how="left"
)


# =========================================================
# 5. CALCULATE CHANGE
# =========================================================

trend["Demand Change"] = (
    trend["Recent Average Demand"]
    - trend["Previous Average Demand"]
)


trend["Demand Change %"] = (
    trend["Demand Change"]
    / trend["Previous Average Demand"].replace(0, pd.NA)
) * 100


# =========================================================
# 6. IDENTIFY DEMAND TREND
# =========================================================

def classify_trend(change_percent):

    if pd.isna(change_percent):
        return "Stable"

    if change_percent > 5:
        return "Increasing"

    elif change_percent < -5:
        return "Decreasing"

    else:
        return "Stable"


trend["Demand Trend"] = (
    trend["Demand Change %"]
    .apply(classify_trend)
)


# =========================================================
# 7. ADD CATEGORY AND REGION
# =========================================================

product_info = (
    df[
        [
            "Store ID",
            "Product ID",
            "Category",
            "Region"
        ]
    ]
    .drop_duplicates(
        subset=["Store ID", "Product ID"]
    )
)


trend = trend.merge(
    product_info,
    on=["Store ID", "Product ID"],
    how="left"
)


# =========================================================
# 8. ROUND VALUES
# =========================================================

trend["Recent Average Demand"] = (
    trend["Recent Average Demand"].round(2)
)

trend["Previous Average Demand"] = (
    trend["Previous Average Demand"].round(2)
)

trend["Demand Change"] = (
    trend["Demand Change"].round(2)
)

trend["Demand Change %"] = (
    trend["Demand Change %"].round(2)
)


# =========================================================
# 9. REORDER COLUMNS
# =========================================================

trend = trend[
    [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Previous Average Demand",
        "Recent Average Demand",
        "Demand Change",
        "Demand Change %",
        "Demand Trend"
    ]
]


# =========================================================
# 10. SAVE RESULT
# =========================================================

output_path = (
    "data/processed/demand_trend_analysis.csv"
)

trend.to_csv(
    output_path,
    index=False
)


# =========================================================
# 11. DISPLAY RESULTS
# =========================================================

print("\n")
print("=" * 70)
print("DEMAND TREND ANALYSIS")
print("=" * 70)

print(
    trend.head(20).to_string(index=False)
)


print("\nTrend Summary:")
print(
    trend["Demand Trend"].value_counts()
)


print("\nDemand trend analysis completed successfully!")

print(
    f"Saved to: {output_path}"
)