import os

# path to your dataset (adjust if needed)
dataset_path = "data/frames"  

real_path = os.path.join(dataset_path, "real")
fake_path = os.path.join(dataset_path, "fake")

# count number of images in each folder
real_count = sum(len(files) for _, _, files in os.walk(real_path))
fake_count = sum(len(files) for _, _, files in os.walk(fake_path))

print(f"Number of real frames: {real_count}")
print(f"Number of fake frames: {fake_count}")

if real_count == fake_count:
    print("✅ Dataset is balanced.")
else:
    print("⚠️ Dataset is NOT balanced.")
    diff = abs(real_count - fake_count)
    print(f"Difference: {diff} frames")
