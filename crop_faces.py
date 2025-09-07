import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

INPUT_DIR = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\frames"
OUTPUT_DIR = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = MTCNN()

for label in ["real", "fake"]:
    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_path, exist_ok=True)

    # loop through subfolders (each video folder)
    for subdir in os.listdir(input_path):
        subfolder_in = os.path.join(input_path, subdir)
        subfolder_out = os.path.join(output_path, subdir)
        os.makedirs(subfolder_out, exist_ok=True)

        for file in tqdm(os.listdir(subfolder_in), desc=f"Processing {label}/{subdir}"):
            img_path = os.path.join(subfolder_in, file)
            save_path = os.path.join(subfolder_out, file)

            # ✅ Skip if face already extracted
            if os.path.exists(save_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = detector.detect_faces(img)
            if faces:
                # take the largest detected face
                faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
                x, y, w, h = faces[0]["box"]
                x, y = max(0, x), max(0, y)
                cropped = img[y:y+h, x:x+w]
                cv2.imwrite(save_path, cropped)
            else:
                # if no face found, skip
                continue
