import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

# Remove rows where lag/rolling features aren't available
required_features = [
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_30"
]

df = df.dropna(subset=required_features).reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

target = "Demand"

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

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# --------------------------------------------------
# 4. PREPROCESSING
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
# 5. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    non_zero = y_test != 0

    mape = np.mean(
        np.abs(
            (y_test[non_zero] - predictions[non_zero])
            / y_test[non_zero]
        )
    ) * 100

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


# --------------------------------------------------
# 6. BASELINE
# --------------------------------------------------

baseline_predictions = X_test["Demand_Lag_1"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

non_zero = y_test != 0

baseline_mape = np.mean(
    np.abs(
        (y_test[non_zero] -
         baseline_predictions[non_zero])
        / y_test[non_zero]
    )
) * 100

baseline_result = {
    "Model": "Naive Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "MAPE": baseline_mape
}


# --------------------------------------------------
# 7. RANDOM FOREST
# --------------------------------------------------

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_result = evaluate_model(
    "Random Forest",
    random_forest,
    X_train,
    X_test,
    y_train,
    y_test
)


# --------------------------------------------------
# 8. GRADIENT BOOSTING
# --------------------------------------------------

gradient_boosting = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

gb_result = evaluate_model(
    "Gradient Boosting",
    gradient_boosting,
    X_train,
    X_test,
    y_train,
    y_test
)


# --------------------------------------------------
# 9. COMPARISON TABLE
# --------------------------------------------------

results = pd.DataFrame([
    baseline_result,
    rf_result,
    gb_result
])

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "MAPE": "{:.2f}%".format
        }
    )
)


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("data/processed/model_comparison.csv")