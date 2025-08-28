import cv2
import os
import glob

# Input video directories
video_dirs = [
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\original",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\DeepFakeDetection",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\Deepfakes",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\Face2Face",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\FaceShifter",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\FaceSwap",
    r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++\NeuralTextures"
]

# Output folder
output_dir = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\frames"
os.makedirs(output_dir, exist_ok=True)

frames_per_video = 10  # number of frames to extract

for video_dir in video_dirs:
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)

        # ✅ Skip if this video already has extracted frames
        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) >= frames_per_video:
            print(f"Skipping {video_name} (already processed).")
            continue

        os.makedirs(video_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            print(f"Warning: {video_path} has no frames.")
            cap.release()
            continue

        step = max(1, frame_count // frames_per_video)
        extracted = 0
        frame_idx = 0

        while cap.isOpened() and extracted < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{extracted}.jpg")
                cv2.imwrite(frame_filename, frame)
                extracted += 1

            frame_idx += 1

        cap.release()
        print(f"Extracted {extracted} frames from {video_name}.")
