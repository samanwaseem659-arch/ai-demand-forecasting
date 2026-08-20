import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

forecast_path = "data/processed/future_demand_forecast.csv"
sales_path = "data/processed/sales_cleaned.csv"
output_path = "data/processed/inventory_analysis.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading future demand forecast...")

forecast = pd.read_csv(forecast_path)

forecast["Date"] = pd.to_datetime(forecast["Date"])

print("Future forecast loaded successfully!")
print("Forecast shape:", forecast.shape)


# Load cleaned sales data to get current inventory information
print("\nLoading inventory data...")

sales = pd.read_csv(sales_path)

sales["Date"] = pd.to_datetime(sales["Date"])

print("Sales data loaded successfully!")
print("Sales shape:", sales.shape)


# ============================================================
# 3. GET LATEST INVENTORY RECORD
# ============================================================

latest_date = sales["Date"].max()

latest_inventory = sales[
    sales["Date"] == latest_date
].copy()

print("\nLatest inventory date:", latest_date)
print(
    "Store/Product combinations:",
    len(latest_inventory)
)


# ============================================================
# 4. SELECT REQUIRED INVENTORY COLUMNS
# ============================================================

inventory_columns = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Ordered",
    "Demand"
]

# Check which columns actually exist
available_columns = [
    col for col in inventory_columns
    if col in latest_inventory.columns
]

latest_inventory = latest_inventory[
    available_columns
].copy()


# ============================================================
# 5. CALCULATE FORECASTED DEMAND
# ============================================================

print("\nCalculating future demand...")

forecast_summary = (
    forecast
    .groupby(["Store ID", "Product ID"])
    .agg(
        Total_Forecast_Demand=(
            "Predicted Demand",
            "sum"
        ),
        Average_Daily_Demand=(
            "Predicted Demand",
            "mean"
        ),
        Maximum_Daily_Demand=(
            "Predicted Demand",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# 6. MERGE INVENTORY + FORECAST
# ============================================================

analysis = pd.merge(
    latest_inventory,
    forecast_summary,
    on=["Store ID", "Product ID"],
    how="left"
)


# ============================================================
# 7. HANDLE MISSING FORECAST VALUES
# ============================================================

analysis[
    [
        "Total_Forecast_Demand",
        "Average_Daily_Demand",
        "Maximum_Daily_Demand"
    ]
] = analysis[
    [
        "Total_Forecast_Demand",
        "Average_Daily_Demand",
        "Maximum_Daily_Demand"
    ]
].fillna(0)


# ============================================================
# 8. SAFETY STOCK
# ============================================================

# Safety stock = 20% of expected 7-day demand

analysis["Safety_Stock"] = (
    analysis["Total_Forecast_Demand"] * 0.20
)


# ============================================================
# 9. REORDER LEVEL
# ============================================================

analysis["Reorder_Level"] = (
    analysis["Total_Forecast_Demand"]
    + analysis["Safety_Stock"]
)


# ============================================================
# 10. REQUIRED INVENTORY
# ============================================================

analysis["Recommended_Inventory"] = (
    analysis["Reorder_Level"]
)


# ============================================================
# 11. INVENTORY SURPLUS / DEFICIT
# ============================================================

analysis["Inventory_Difference"] = (
    analysis["Inventory Level"]
    - analysis["Recommended_Inventory"]
)


# ============================================================
# 12. INVENTORY COVERAGE
# ============================================================

analysis["Inventory_Coverage_Days"] = np.where(
    analysis["Average_Daily_Demand"] > 0,
    analysis["Inventory Level"]
    / analysis["Average_Daily_Demand"],
    999
)


# ============================================================
# 13. STOCKOUT RISK
# ============================================================

def stockout_risk(row):

    inventory = row["Inventory Level"]
    demand = row["Total_Forecast_Demand"]

    if demand <= 0:
        return "Low"

    if inventory < demand * 0.50:
        return "High"

    elif inventory < demand:
        return "Medium"

    else:
        return "Low"


analysis["Stockout_Risk"] = analysis.apply(
    stockout_risk,
    axis=1
)


# ============================================================
# 14. OVERSTOCK RISK
# ============================================================

def overstock_risk(row):

    inventory = row["Inventory Level"]
    demand = row["Total_Forecast_Demand"]

    if demand <= 0:
        if inventory > 0:
            return "High"
        return "Low"

    if inventory > demand * 2:
        return "High"

    elif inventory > demand * 1.5:
        return "Medium"

    else:
        return "Low"


analysis["Overstock_Risk"] = analysis.apply(
    overstock_risk,
    axis=1
)


# ============================================================
# 15. RECOMMENDED ORDER QUANTITY
# ============================================================

analysis["Recommended_Order_Quantity"] = (
    analysis["Recommended_Inventory"]
    - analysis["Inventory Level"]
).clip(lower=0)


# ============================================================
# 16. RECOMMENDED ACTION
# ============================================================

def recommended_action(row):

    inventory = row["Inventory Level"]
    demand = row["Total_Forecast_Demand"]

    # No expected demand
    if demand <= 0:

        if inventory > 0:
            return "Avoid Ordering"

        return "No Action"

    # Severe shortage
    if inventory < demand * 0.50:
        return "Order Immediately"

    # Moderate shortage
    elif inventory < demand:
        return "Reorder"

    # Overstock
    elif inventory > demand * 2:
        return "Reduce Inventory"

    # Slightly high inventory
    elif inventory > demand * 1.5:
        return "Monitor Overstock"

    # Healthy inventory
    else:
        return "Inventory Healthy"


analysis["Recommended_Action"] = analysis.apply(
    recommended_action,
    axis=1
)


# ============================================================
# 17. ROUND NUMERICAL VALUES
# ============================================================

numeric_columns = [
    "Inventory Level",
    "Demand",
    "Total_Forecast_Demand",
    "Average_Daily_Demand",
    "Maximum_Daily_Demand",
    "Safety_Stock",
    "Reorder_Level",
    "Recommended_Inventory",
    "Inventory_Difference",
    "Inventory_Coverage_Days",
    "Recommended_Order_Quantity"
]

for column in numeric_columns:

    if column in analysis.columns:

        analysis[column] = (
            analysis[column]
            .round(2)
        )


# ============================================================
# 18. SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",

    "Inventory Level",
    "Demand",

    "Total_Forecast_Demand",
    "Average_Daily_Demand",
    "Maximum_Daily_Demand",

    "Safety_Stock",
    "Reorder_Level",
    "Recommended_Inventory",

    "Inventory_Difference",
    "Inventory_Coverage_Days",

    "Recommended_Order_Quantity",

    "Stockout_Risk",
    "Overstock_Risk",

    "Recommended_Action"
]

# Keep only columns that exist
final_columns = [
    col
    for col in final_columns
    if col in analysis.columns
]

analysis = analysis[final_columns]


# ============================================================
# 19. SAVE INVENTORY ANALYSIS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

analysis.to_csv(
    output_path,
    index=False
)


# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("INVENTORY ANALYSIS COMPLETED")
print("==========================================")

print("\nRecords analyzed:")
print(len(analysis))

print("\nStockout Risk:")
print(
    analysis["Stockout_Risk"]
    .value_counts()
)

print("\nOverstock Risk:")
print(
    analysis["Overstock_Risk"]
    .value_counts()
)

print("\nRecommended Actions:")
print(
    analysis["Recommended_Action"]
    .value_counts()
)

print("\nSample Inventory Analysis:")
print(
    analysis.head(20).to_string(index=False)
)

print("\nInventory analysis saved successfully!")
print(f"Location: {output_path}")