import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch
import random
import matplotlib.pyplot as plt

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
mtcnn = MTCNN(image_size=224, margin=40, device=device)  # 224x224 crops

# Input/output paths
frames_dir = "data/frames"
cropped_dir = "data/cropped"

os.makedirs(cropped_dir, exist_ok=True)

total_saved = 0
total_skipped = 0

for label in ["real", "fake"]:
    input_path = os.path.join(frames_dir, label)
    output_path = os.path.join(cropped_dir, label)
    os.makedirs(output_path, exist_ok=True)

    videos = os.listdir(input_path)

    for video in tqdm(videos, desc=f"Processing {label}"):
        video_in_path = os.path.join(input_path, video)
        video_out_path = os.path.join(output_path, video)
        os.makedirs(video_out_path, exist_ok=True)

        frames = os.listdir(video_in_path)

        for frame in frames:
            frame_path = os.path.join(video_in_path, frame)
            save_path = os.path.join(video_out_path, frame)

            try:
                img = Image.open(frame_path)

                # Detect face
                face = mtcnn(img)

                if face is not None:
                    # Convert from [-1,1] → [0,255]
                    face = (face.permute(1, 2, 0).numpy() + 1) / 2 * 255
                    face = face.astype('uint8')
                    Image.fromarray(face).save(save_path)
                    total_saved += 1
                else:
                    total_skipped += 1

            except Exception as e:
                print(f"❌ Error processing {frame_path}: {e}")
                total_skipped += 1

print(f"\n✅ Face cropping completed. Saved: {total_saved}, Skipped: {total_skipped}")

# --- Quick Preview of some cropped faces ---
print("\n🔍 Showing a preview of random cropped faces...")
sample_images = []
for label in ["real", "fake"]:
    path = os.path.join(cropped_dir, label)
    videos = os.listdir(path)
    if not videos:
        continue
    video = random.choice(videos)
    frames = os.listdir(os.path.join(path, video))
    if frames:
        sample_images.append(os.path.join(path, video, random.choice(frames)))

# Plot sample faces
if sample_images:
    plt.figure(figsize=(10, 5))
    for i, img_path in enumerate(sample_images[:6]):  # show up to 6
        img = Image.open(img_path)
        plt.subplot(1, len(sample_images), i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
else:
    print("⚠️ No cropped images found to preview.")
