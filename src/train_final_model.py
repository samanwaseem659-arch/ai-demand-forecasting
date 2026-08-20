import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD FEATURE DATA
# ============================================================

print("Loading engineered feature dataset...")

df = pd.read_csv(
    "data/processed/features.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(
    subset=["Date"]
).copy()

# Sort chronologically by store/product/date
df = (
    df.sort_values(
        ["Store ID", "Product ID", "Date"]
    )
    .reset_index(drop=True)
)

print(
    f"Feature dataset loaded successfully: {df.shape}"
)


# ============================================================
# 2. REQUIRED ENGINEERED FEATURES
# ============================================================

required_features = [
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",

    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    "Quarter",
    "IsWeekend",
    "Price_Change"
]


# ============================================================
# 3. REMOVE ROWS WITHOUT REQUIRED FEATURES
# ============================================================

before_rows = len(df)

df = df.dropna(
    subset=required_features
).reset_index(drop=True)

after_rows = len(df)

print(
    f"Rows removed because of missing engineered features: "
    f"{before_rows - after_rows:,}"
)

print(
    f"Rows available for training: {after_rows:,}"
)


# ============================================================
# 4. TARGET
# ============================================================

target = "Demand"


# ============================================================
# 5. MODEL FEATURES
# ============================================================

features = [

    # ----------------------------------------
    # BUSINESS / INVENTORY FEATURES
    # ----------------------------------------

    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",

    # ----------------------------------------
    # CALENDAR FEATURES
    # ----------------------------------------

    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "Quarter",
    "IsWeekend",

    # ----------------------------------------
    # DEMAND LAG FEATURES
    # ----------------------------------------

    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",

    # ----------------------------------------
    # ROLLING DEMAND FEATURES
    # ----------------------------------------

    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    # ----------------------------------------
    # PRICE CHANGE
    # ----------------------------------------

    "Price_Change",

    # ----------------------------------------
    # CATEGORICAL FEATURES
    # ----------------------------------------

    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

missing_features = [
    col
    for col in features + [target]
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# 7. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 8. CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"

]


# ============================================================
# 9. NUMERICAL FEATURES
# ============================================================

numerical_features = [

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

    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",

    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    "Price_Change"

]


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )

    ]
)


# ============================================================
# 11. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)


# ============================================================
# 12. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]

)


# ============================================================
# 13. TRAIN FINAL MODEL
# ============================================================

print()
print("============================================")
print("Training final Random Forest model...")
print("============================================")

pipeline.fit(
    X,
    y
)

print(
    "Final model training completed!"
)


# ============================================================
# 14. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

model_path = (
    "models/random_forest.pkl"
)

joblib.dump(
    pipeline,
    model_path
)

print()
print(
    "Model saved successfully!"
)

print(
    f"Location: {model_path}"
)


# ============================================================
# 16. SAVE MODEL METADATA
# ============================================================

metadata = {

    "target": target,

    "features": features,

    "categorical_features":
        categorical_features,

    "numerical_features":
        numerical_features,

    "engineered_features":
        required_features,

    "model":
        "RandomForestRegressor",

    "n_estimators":
        100,

    "random_state":
        42

}


metadata_path = (
    "models/model_metadata.pkl"
)

joblib.dump(
    metadata,
    metadata_path
)

print(
    "Model metadata saved successfully!"
)

print(
    f"Location: {metadata_path}"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print()
print("============================================")
print("FINAL MODEL SUMMARY")
print("============================================")

print(
    f"Training rows: {len(X):,}"
)

print(
    f"Number of features: {len(features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)

print(
    f"Numerical features: "
    f"{len(numerical_features)}"
)

print(
    "Model: Random Forest Regressor"
)

print(
    "Training completed successfully!"
)

print("============================================")