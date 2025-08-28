import pandas as pd

# Path to your CSV file (update if needed)
csv_path = r"D:\deepfake-detector\faceforensic++\csv\Deepfakes.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Show basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n")
print(df.head())

# Show column names
print("\nColumns:", df.columns.tolist())
