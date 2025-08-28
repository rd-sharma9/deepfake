import os
import pandas as pd

# Paths
base_path = r"D:\deepfake-detector\faceforensic++\csv"
csv_files = [
    "DeepFakeDetection.csv",
    "Deepfakes.csv",
    "Face2Face.csv",
    "FaceShifter.csv",
    "FaceSwap.csv",
    "NeuralTextures.csv",
    "original.csv"
]

cleaned_dfs = []

for file in csv_files:
    file_path = os.path.join(base_path, file)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Remove unnecessary unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Add Source column from filename
    source_name = os.path.splitext(file)[0]  # e.g., "Deepfakes"
    df["Source"] = source_name
    
    # Save cleaned version
    cleaned_path = os.path.join(base_path, f"{source_name}_clean.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned {file} → {source_name}_clean.csv")
    
    cleaned_dfs.append(df)

# Merge all into one
if cleaned_dfs:
    merged_df = pd.concat(cleaned_dfs, ignore_index=True)
    merged_path = os.path.join(base_path, "all_data.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"\n📂 All datasets merged into {merged_path}")
    print(f"Total samples: {len(merged_df)}")
    print("Columns:", list(merged_df.columns))
