import pandas as pd


def load_data():
    file_path = "data/raw/sales_data.csv"

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_data()