import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path where your combined CSV will be saved
output_csv = r"D:\deepfake-detector\all_data.csv"

# Load your original dataset CSV (if already created)
df = pd.read_csv(output_csv)

print(f"✅ Loaded {len(df)} samples")
print(df['Label'].value_counts())

# --- Split the dataset ---
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

print("\n📊 Split Summary:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# --- Save CSVs ---
base_dir = os.path.dirname(output_csv)

train_path = os.path.join(base_dir, "train.csv")
val_path = os.path.join(base_dir, "val.csv")
test_path = os.path.join(base_dir, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\n✅ Train/Val/Test CSVs created at {base_dir}")
