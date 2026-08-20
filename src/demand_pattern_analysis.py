import pandas as pd


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/processed/sales_cleaned.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Records:", len(df))


# =========================================================
# 2. CREATE TIME FEATURES
# =========================================================

df["DayOfWeek"] = df["Date"].dt.day_name()

df["Month"] = df["Date"].dt.month

df["MonthName"] = df["Date"].dt.month_name()

df["Year"] = df["Date"].dt.year


# =========================================================
# 3. WEEKLY DEMAND PATTERN
# =========================================================

weekly_pattern = (
    df.groupby("DayOfWeek")["Demand"]
    .mean()
    .reset_index()
)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_pattern["DayOfWeek"] = pd.Categorical(
    weekly_pattern["DayOfWeek"],
    categories=day_order,
    ordered=True
)

weekly_pattern = (
    weekly_pattern
    .sort_values("DayOfWeek")
    .reset_index(drop=True)
)

weekly_pattern["Average Demand"] = (
    weekly_pattern["Demand"].round(2)
)

weekly_pattern = weekly_pattern[
    ["DayOfWeek", "Average Demand"]
]


# =========================================================
# 4. MONTHLY DEMAND PATTERN
# =========================================================

monthly_pattern = (
    df.groupby(
        ["Month", "MonthName"]
    )["Demand"]
    .mean()
    .reset_index()
)

monthly_pattern["Average Demand"] = (
    monthly_pattern["Demand"].round(2)
)

monthly_pattern = monthly_pattern[
    [
        "Month",
        "MonthName",
        "Average Demand"
    ]
].sort_values("Month")


# =========================================================
# 5. SEASONAL DEMAND PATTERN
# =========================================================

seasonal_pattern = (
    df.groupby("Seasonality")["Demand"]
    .mean()
    .reset_index()
)

seasonal_pattern["Average Demand"] = (
    seasonal_pattern["Demand"].round(2)
)

seasonal_pattern = seasonal_pattern[
    [
        "Seasonality",
        "Average Demand"
    ]
].sort_values(
    "Average Demand",
    ascending=False
)


# =========================================================
# 6. HIGH-DEMAND PERIODS
# =========================================================

high_threshold = df["Demand"].quantile(0.90)

high_demand = df[
    df["Demand"] >= high_threshold
].copy()


high_demand_summary = (
    high_demand.groupby(
        "MonthName"
    )["Demand"]
    .mean()
    .reset_index()
)

high_demand_summary["Average Demand"] = (
    high_demand_summary["Demand"].round(2)
)

high_demand_summary = (
    high_demand_summary[
        [
            "MonthName",
            "Average Demand"
        ]
    ]
    .sort_values(
        "Average Demand",
        ascending=False
    )
)


# =========================================================
# 7. LOW-DEMAND PERIODS
# =========================================================

low_threshold = df["Demand"].quantile(0.10)

low_demand = df[
    df["Demand"] <= low_threshold
].copy()


low_demand_summary = (
    low_demand.groupby(
        "MonthName"
    )["Demand"]
    .mean()
    .reset_index()
)

low_demand_summary["Average Demand"] = (
    low_demand_summary["Demand"].round(2)
)

low_demand_summary = (
    low_demand_summary[
        [
            "MonthName",
            "Average Demand"
        ]
    ]
    .sort_values(
        "Average Demand"
    )
)


# =========================================================
# 8. OVERALL PATTERN SUMMARY
# =========================================================

highest_day = weekly_pattern.loc[
    weekly_pattern["Average Demand"].idxmax()
]

lowest_day = weekly_pattern.loc[
    weekly_pattern["Average Demand"].idxmin()
]


highest_month = monthly_pattern.loc[
    monthly_pattern["Average Demand"].idxmax()
]

lowest_month = monthly_pattern.loc[
    monthly_pattern["Average Demand"].idxmin()
]


highest_season = seasonal_pattern.loc[
    seasonal_pattern["Average Demand"].idxmax()
]

lowest_season = seasonal_pattern.loc[
    seasonal_pattern["Average Demand"].idxmin()
]


# =========================================================
# 9. SAVE INDIVIDUAL ANALYSES
# =========================================================

weekly_pattern.to_csv(
    "data/processed/weekly_demand_pattern.csv",
    index=False
)

monthly_pattern.to_csv(
    "data/processed/monthly_demand_pattern.csv",
    index=False
)

seasonal_pattern.to_csv(
    "data/processed/seasonal_demand_pattern.csv",
    index=False
)

high_demand_summary.to_csv(
    "data/processed/high_demand_periods.csv",
    index=False
)

low_demand_summary.to_csv(
    "data/processed/low_demand_periods.csv",
    index=False
)


# =========================================================
# 10. DISPLAY IMPORTANT PATTERNS
# =========================================================

print("\n")
print("=" * 70)
print("IMPORTANT DEMAND PATTERNS")
print("=" * 70)


print("\nWeekly Pattern:")
print(
    weekly_pattern.to_string(
        index=False
    )
)


print("\nMonthly Pattern:")
print(
    monthly_pattern.to_string(
        index=False
    )
)


print("\nSeasonal Pattern:")
print(
    seasonal_pattern.to_string(
        index=False
    )
)


print("\nHigh-Demand Periods:")
print(
    high_demand_summary.head(10).to_string(
        index=False
    )
)


print("\nLow-Demand Periods:")
print(
    low_demand_summary.head(10).to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("KEY DEMAND PATTERNS")
print("=" * 70)


print(
    f"\nHighest-demand day: "
    f"{highest_day['DayOfWeek']} "
    f"({highest_day['Average Demand']:.2f})"
)


print(
    f"Lowest-demand day: "
    f"{lowest_day['DayOfWeek']} "
    f"({lowest_day['Average Demand']:.2f})"
)


print(
    f"Highest-demand month: "
    f"{highest_month['MonthName']} "
    f"({highest_month['Average Demand']:.2f})"
)


print(
    f"Lowest-demand month: "
    f"{lowest_month['MonthName']} "
    f"({lowest_month['Average Demand']:.2f})"
)


print(
    f"Highest-demand season: "
    f"{highest_season['Seasonality']} "
    f"({highest_season['Average Demand']:.2f})"
)


print(
    f"Lowest-demand season: "
    f"{lowest_season['Seasonality']} "
    f"({lowest_season['Average Demand']:.2f})"
)


print("\nDemand pattern analysis completed successfully!")