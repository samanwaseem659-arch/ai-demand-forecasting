import pandas as pd


def preprocess_data():

    # Load raw dataset
    file_path = "data/raw/sales_data.csv"
    df = pd.read_csv(file_path)

    print("Original shape:", df.shape)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort data by date
    df = df.sort_values("Date")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Save processed dataset
    output_path = "data/processed/sales_cleaned.csv"
    df.to_csv(output_path, index=False)

    print("\nProcessed data saved successfully!")
    print("New shape:", df.shape)


if __name__ == "__main__":
    preprocess_data()