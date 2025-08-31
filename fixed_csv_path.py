import pandas as pd
import glob
import os

# Folder where your CSVs are stored
csv_folder = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector"

# Find all CSV files in root folder
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

for csv_file in csv_files:
    print(f"Processing {csv_file} ...")
    df = pd.read_csv(csv_file)

    # Fix paths: replace D:\ with C:\ (pointing to your OneDrive location)
    df.iloc[:, 0] = df.iloc[:, 0].str.replace(
        r"D:\\deepfake-detector",
        r"C:\\Users\\ASUS\\OneDrive\\Desktop\\deepfake-detector",
        regex=True
    )

    # Save back
    df.to_csv(csv_file, index=False)
    print(f"✅ Fixed paths in {csv_file}")
