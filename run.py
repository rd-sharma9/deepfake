import os

dataset_path = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\frames"

real_path = os.path.join(dataset_path, "real")
fake_path = os.path.join(dataset_path, "fake")

# Count frames inside all subfolders
real_count = sum([len(files) for r, d, files in os.walk(real_path)])
fake_count = sum([len(files) for r, d, files in os.walk(fake_path)])

print(f"Total REAL frames: {real_count}")
print(f"Total FAKE frames: {fake_count}")
