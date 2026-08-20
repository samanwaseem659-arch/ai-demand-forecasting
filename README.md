# AI Demand Forecasting & Smart Inventory Optimization Platform

An end-to-end machine learning platform for **demand forecasting and intelligent inventory analysis**.

The system uses historical sales, inventory, pricing, promotion, weather, seasonality, and other business variables to predict future product demand and support better inventory decisions.

---

## 🚀 Project Overview

Inventory management is a major challenge for businesses.

Too much inventory can increase:

* Storage costs
* Capital tied up in stock
* Product waste
* Operational costs

Too little inventory can cause:

* Stockouts
* Lost sales
* Customer dissatisfaction
* Supply-chain disruption

This project addresses these challenges by combining **data preprocessing, demand analysis, machine learning, future forecasting, and inventory analysis** into a single platform.

### Main Goal

> Predict future product demand and use those predictions to support smarter inventory-management decisions.

---

## 🏭 Industry Use Case

This platform can be used in industries where accurate demand forecasting
and inventory planning are important, including:

* Retail
* E-commerce
* Grocery and supermarkets
* Consumer goods
* Warehousing and distribution
* Manufacturing
* Pharmaceutical inventory management

The system helps businesses estimate future demand and identify potential
stockout and overstock conditions.

---
## 🎯 Project Objectives

The main objectives of the project are:

* Predict future product demand using machine learning.
* Analyze historical demand trends and patterns.
* Compare different forecasting models.
* Identify products with increasing or decreasing demand.
* Detect potential stockout and overstock risks.
* Estimate appropriate inventory levels.
* Provide inventory recommendations based on forecasted demand.
* Present results through an interactive Streamlit dashboard.

---

## ✨ Key Features

### 📊 Data Processing

* Historical sales-data loading
* Data validation
* Data cleaning
* Missing-value handling
* Date processing
* Numeric-value validation

### 📈 Demand Analysis

* Demand trend analysis
* Demand pattern analysis
* Product-level demand analysis
* Time-based analysis
* Seasonal pattern analysis

### 🤖 Machine Learning

* Naive baseline
* Random Forest
* Gradient Boosting
* Model comparison
* MAE evaluation
* RMSE evaluation
* MAPE evaluation
* Final model selection

### 🔮 Demand Forecasting

* Future demand prediction
* Product-level forecasting
* Multi-day future forecasts
* Saved forecast results

### 📦 Inventory Intelligence

* Inventory-level analysis
* Stockout-risk identification
* Overstock-risk identification
* Safety-stock analysis
* Inventory recommendations

### 🖥️ Interactive Dashboard

The Streamlit dashboard provides an interactive interface for viewing:

* Sales data
* Demand trends
* Demand patterns
* Forecast results
* Inventory analysis
* Model performance
* Predictions

---

## 🏗️ Complete System Workflow

```text
Historical Sales Data
        │
        ▼
Data Validation
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Demand Trend Analysis
        │
        ▼
Demand Pattern Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Comparison
        │
        ▼
Final Model Training
        │
        ▼
Demand Prediction
        │
        ▼
Future Demand Forecast
        │
        ▼
Inventory Analysis
        │
        ▼
Streamlit Dashboard
```

---

## 📂 Project Structure

```text
AI demand forecasting/
│
├── app/
│
├── assets/
│   └── screenshots/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── project_documentation.md
│
├── models/
│   └── random_forest.pkl
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── pre_process.py
│   ├── demand_trend_analysis.py
│   ├── demand_pattern_analysis.py
│   ├── feature_eng.py
│   ├── forecast_model.py
│   ├── train_final_model.py
│   ├── predict.py
│   ├── future_forecast.py
│   ├── inventory_analysis.py
│   └── model_comparison.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── SKILL.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 📋 Dataset

The project uses historical business and sales data.

The dataset contains variables such as:

| Column               | Description                       |
| -------------------- | --------------------------------- |
| `Date`               | Date of the sales record          |
| `Store ID`           | Store identifier                  |
| `Product ID`         | Product identifier                |
| `Category`           | Product category                  |
| `Region`             | Geographic region                 |
| `Inventory Level`    | Current inventory level           |
| `Units Sold`         | Number of units sold              |
| `Units Ordered`      | Number of units ordered           |
| `Price`              | Product price                     |
| `Discount`           | Applied discount                  |
| `Weather Condition`  | Weather information               |
| `Promotion`          | Promotion indicator               |
| `Competitor Pricing` | Competitor price                  |
| `Seasonality`        | Seasonal information              |
| `Epidemic`           | Epidemic-related indicator        |
| `Demand`             | Demand value used for forecasting |

---

## 🧹 Data Preprocessing

Before machine learning, the raw data passes through a preprocessing pipeline.

The preprocessing stage handles:

* Date conversion
* Missing values
* Invalid values
* Data-type conversion
* Duplicate records
* Required-column validation
* Feature preparation

The cleaned dataset is saved as:

```text
data/processed/sales_cleaned.csv
```

---

## 📊 Demand Analysis

### Demand Trend Analysis

The system analyzes demand over time to identify:

* Increasing demand
* Decreasing demand
* Stable demand
* Long-term demand movement

Implemented in:

```text
src/demand_trend_analysis.py
```

### Demand Pattern Analysis

The system examines recurring demand behavior including:

* Seasonal patterns
* Promotional effects
* Product-level differences
* Store-level differences
* Regional variations
* Time-based demand behavior

Implemented in:

```text
src/demand_pattern_analysis.py
```

---

## ⚙️ Feature Engineering

Feature engineering converts historical business information into machine-learning features.

Features may include:

* Time-based features
* Historical demand features
* Lag features
* Rolling statistics
* Product information
* Store information
* Pricing information
* Promotion information
* Seasonal information

Implemented in:

```text
src/feature_eng.py
```

---

# 🤖 Machine Learning

The project compares multiple forecasting approaches.

## Models

### 1. Naive Baseline

Provides a simple reference point against which machine-learning models can be evaluated.

### 2. Random Forest

A tree-based ensemble model used for demand prediction.

### 3. Gradient Boosting

A sequential ensemble-learning approach used as an alternative forecasting model.

---

## 📈 Model Performance

The models were evaluated using three metrics:

* MAE
* RMSE
* MAPE

### Results

| Model             |       MAE |      RMSE |       MAPE |
| ----------------- | --------: | --------: | ---------: |
| Naive Baseline    |     40.03 |     51.79 |     52.84% |
| Random Forest     | **16.51** | **22.64** | **24.35%** |
| Gradient Boosting |     20.03 |     26.10 |     30.01% |

### Best Model

**Random Forest**

Random Forest achieved the lowest MAE, RMSE, and MAPE among the evaluated models.

The model comparison results are stored in:

```text
data/processed/model_comparison.csv
```

---

## 🏆 Final Model

After model comparison, the Random Forest model was selected as the final forecasting model.

The trained model is saved as:

```text
models/random_forest.pkl
```

The model can be loaded later for predictions without retraining.

The final-model training process is handled by:

```text
src/train_final_model.py
```

---

## 🔮 Future Demand Forecasting

The trained Random Forest model is used to generate future demand predictions.

The forecasting process:

1. Loads the trained model.
2. Loads the required data.
3. Creates the required features.
4. Generates future prediction dates.
5. Predicts product demand.
6. Saves the forecast results.

Implemented in:

```text
src/future_forecast.py
```

Output:

```text
data/processed/future_demand_forecast.csv
```

---

## 📦 Inventory Analysis

The inventory-analysis stage combines forecasted demand with inventory information.

It helps identify products that may require attention.

Potential inventory conditions include:

### 🔴 Stockout Risk

Demand may exceed available inventory.

### 🟡 Overstock Risk

Inventory may be significantly higher than expected demand.

### 🟢 Suitable Inventory

Current inventory may be sufficient for expected demand.

The system can also support:

* Safety-stock analysis
* Recommended inventory levels
* Inventory-risk identification
* Demand-based inventory planning

Implemented in:

```text
src/inventory_analysis.py
```

Output:

```text
data/processed/inventory_analysis.csv
```

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to explore the results of the complete forecasting pipeline.

### Dashboard Areas

* Sales Data
* Demand Trend Analysis
* Demand Pattern Analysis
* Future Demand Forecast
* Inventory Analysis
* Model Performance
* Predictions

The dashboard converts the underlying machine-learning results into a user-friendly interface.

---

## 📸 Dashboard Screenshots

Dashboard screenshots can be placed in:

```text
assets/screenshots/
```

Recommended screenshots include:

```text
assets/screenshots/
├── dashboard.png
├── demand-patterns.png
├── forecasting.png
├── inventory-analysis.png
└── model-performance.png
```

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data processing           |
| NumPy        | Numerical computation     |
| Scikit-learn | Machine learning          |
| Joblib       | Model serialization       |
| Streamlit    | Interactive dashboard     |
| Matplotlib   | Visualization             |
| Pytest       | Testing                   |
| Git          | Version control           |
| GitHub       | Repository hosting        |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/samanwaseem659-arch/ai-demand-forecasting.git
```

Move into the project directory:

```bash
cd AI-demand-forecasting
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run the Streamlit Dashboard

From the project root:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 🔬 Running the Tests

Run:

```bash
pytest tests/
```

The test suite validates important project artifacts including:

* Processed data
* Required columns
* Model availability
* Model comparison results
* Future forecast results
* Inventory analysis results

---

# 📁 Important Output Files

| File                         | Purpose                             |
| ---------------------------- | ----------------------------------- |
| `sales_cleaned.csv`          | Cleaned historical sales data       |
| `model_comparison.csv`       | Model evaluation results            |
| `future_demand_forecast.csv` | Future demand predictions           |
| `inventory_analysis.csv`     | Inventory recommendations and risks |
| `random_forest.pkl`          | Trained final Random Forest model   |

---

# 🧪 Testing

The project includes automated tests located in:

```text
tests/test_pipeline.py
```

The tests verify that important pipeline outputs are available and readable.

Run:

```bash
pytest tests/
```

---

# 📚 Documentation

Additional technical documentation is available in:

```text
docs/project_documentation.md
```

The project's technical skills and workflow are documented in:

```text
SKILL.md
```

---

# 🔐 License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

---

# 🔮 Future Improvements

Possible future improvements include:

* XGBoost forecasting
* LightGBM forecasting
* Advanced time-series models
* LSTM/GRU forecasting
* Automated hyperparameter optimization
* Real-time inventory monitoring
* Automated reorder alerts
* Database integration
* Cloud deployment
* Automated model retraining
* Advanced anomaly detection
* Real-time business dashboards

---

# 🚀 Deployment

The application is developed using Streamlit and can be run locally or deployed to a cloud-based application hosting platform.

## Local Deployment

After installing the project dependencies, run:

```bash
streamlit run app.py
```

The application will open in a web browser and provide access to the interactive demand forecasting and inventory optimization dashboard.

## Deployment Status

The project has been tested successfully with local Streamlit execution.

Cloud deployment can be added using a platform that supports Streamlit applications.

---

# 🤗 Hugging Face Deployment

The project is designed to support deployment through Hugging Face Spaces using Streamlit.

A Hugging Face deployment can use:

* `app.py` as the main application
* `requirements.txt` for Python dependencies
* `assets/` for screenshots and visual assets
* `src/` for data processing and machine-learning modules
* `models/` for the trained forecasting model

## Current Status

The project is prepared for Hugging Face deployment but is **not currently deployed to a Hugging Face Space**.

**Hugging Face Space:** Not deployed yet.

---

# 🐙 GitHub

The complete project source code and documentation are maintained in the GitHub repository.

**Repository:**

https://github.com/samanwaseem659-arch/ai-demand-forecasting

The repository contains:

* Streamlit application
* Data-processing pipeline
* Machine-learning modules
* Trained forecasting model
* Forecasting outputs
* Inventory-analysis outputs
* Documentation
* Tests
* Screenshots
* Project license
* Dependency configuration

Git Large File Storage (Git LFS) is used for the large trained model artifact.

---
---

# ⚠️ Limitations

The current version of the platform has the following limitations:

* Forecast accuracy depends on the quality and quantity of historical data.
* The current system primarily uses Random Forest for final demand forecasting.
* Advanced deep-learning time-series models such as LSTM and GRU are not currently implemented.
* Real-time external market and competitor data integration is not currently available.
* Inventory recommendations depend on the available inventory and demand information.
* Automated real-time inventory monitoring is not currently implemented.
* Automated reorder alerts are not currently implemented.
* The application has been tested with local Streamlit execution.
* Hugging Face cloud deployment has been prepared but has not yet been completed.

---

# 🎯 Project Outcome

This project demonstrates a complete machine-learning solution for a real-world business problem.

The platform transforms:

```text
Historical Data
      ↓
Data Processing
      ↓
Business Insights
      ↓
Machine Learning
      ↓
Demand Predictions
      ↓
Inventory Intelligence
      ↓
Better Inventory Decisions
```

The ultimate objective is to help businesses make more informed inventory decisions while reducing the risk of **stockouts and overstocking**.

---

## 👩‍💻 Author

**Saman Waseem**

AI Demand Forecasting & Smart Inventory Optimization Platform

