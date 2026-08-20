import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. LOAD FEATURE DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/features.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. REMOVE ROWS WITH MISSING LAG/ROLLING VALUES
# --------------------------------------------------

required_features = [
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30"
]

df = df.dropna(subset=required_features)

print("Shape after removing unavailable lag values:", df.shape)


# --------------------------------------------------
# 3. DEFINE TARGET
# --------------------------------------------------

target = "Demand"


# --------------------------------------------------
# 4. DEFINE FEATURES
# --------------------------------------------------

features = [
    # Numerical features
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",

    # Time features
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",

    # Historical demand features
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30",

    # Categorical features
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]


X = df[features]
y = df[target]


# --------------------------------------------------
# 5. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

# Use the first 80% of time for training
# and the final 20% for testing.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining period:")
print(df["Date"].iloc[:split_index].min(), "to",
      df["Date"].iloc[:split_index].max())

print("\nTesting period:")
print(
    df["Date"].iloc[split_index:].min(),
    "to",
    df["Date"].iloc[split_index:].max()
)


# --------------------------------------------------
# 6. DEFINE FEATURE TYPES
# --------------------------------------------------

categorical_features = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]

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
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30"
]


# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 8. RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 9. CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 10. TRAIN
# --------------------------------------------------

print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 11. PREDICT
# --------------------------------------------------

predictions = pipeline.predict(X_test)


# --------------------------------------------------
# 12. EVALUATE
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)


# MAPE
non_zero = y_test != 0

mape = np.mean(
    np.abs(
        (y_test[non_zero] - predictions[non_zero])
        / y_test[non_zero]
    )
) * 100


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")