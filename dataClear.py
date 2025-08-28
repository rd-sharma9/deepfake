# organize_frames.py
import os
import glob
import shutil

# ---- EDIT THESE IF YOUR PATHS DIFFER ----
DATASET_ROOT = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\faceforensic++"
FRAMES_DIR    = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\frames"
# -----------------------------------------

REAL_HINTS = ["original", os.sep + "real" + os.sep, os.sep + "pristine" + os.sep]
FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
FAKE_HINTS = ["manipulated", os.sep + "fake" + os.sep] + [f.lower() for f in FAKE_FOLDERS]

def infer_label_from_path(p):
    lp = p.lower()
    # Real?
    if any(h in lp for h in REAL_HINTS):
        return "real"
    # Fake?
    if any(h in lp for h in FAKE_HINTS):
        return "fake"
    # Special case: DeepFakeDetection may contain real/fake; try to infer
    if "deepfakedetection" in lp:
        if any(h in lp for h in REAL_HINTS):
            return "real"
        if any(h in lp for h in FAKE_HINTS):
            return "fake"
        # If unknown inside DeepFakeDetection, treat as fake by default
        return "fake"
    return None

def build_label_map():
    """Scan dataset root and map base video names to labels."""
    video_paths = glob.glob(os.path.join(DATASET_ROOT, "**", "*.mp4"), recursive=True)
    label_map = {}
    for vp in video_paths:
        base = os.path.splitext(os.path.basename(vp))[0]
        label = infer_label_from_path(vp)
        if label is None:
            continue
        # In case of duplicates, prefer explicit original over fake
        if base in label_map:
            if label_map[base] == "fake" and label == "real":
                label_map[base] = "real"
        else:
            label_map[base] = label
    return label_map

def safe_merge_move(src_dir, dst_dir):
    """Move a directory; if destination exists, merge files."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isdir(s):
            safe_merge_move(s, d)
        else:
            # If a file with same name exists, add a suffix
            if os.path.exists(d):
                root, ext = os.path.splitext(name)
                i = 1
                while os.path.exists(os.path.join(dst_dir, f"{root}__{i}{ext}")):
                    i += 1
                d = os.path.join(dst_dir, f"{root}__{i}{ext}")
            shutil.move(s, d)
    # Remove src if empty
    try:
        os.rmdir(src_dir)
    except OSError:
        pass

def move_stray_images_to_misc():
    misc_dir = os.path.join(FRAMES_DIR, "misc")
    os.makedirs(misc_dir, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    moved = 0
    for name in os.listdir(FRAMES_DIR):
        p = os.path.join(FRAMES_DIR, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            shutil.move(p, os.path.join(misc_dir, name))
            moved += 1
    return moved

if __name__ == "__main__":
    os.makedirs(FRAMES_DIR, exist_ok=True)
    real_root = os.path.join(FRAMES_DIR, "real")
    fake_root = os.path.join(FRAMES_DIR, "fake")
    os.makedirs(real_root, exist_ok=True)
    os.makedirs(fake_root, exist_ok=True)

    label_map = build_label_map()
    print(f"Mapped {len(label_map)} video names from dataset.")

    # Clean up any loose images at root
    stray = move_stray_images_to_misc()
    if stray:
        print(f"Moved {stray} stray image(s) to 'misc/'.")

    moved_real = moved_fake = skipped_known = 0
    unmatched = []

    # Go over every top-level item in FRAMES_DIR
    for name in os.listdir(FRAMES_DIR):
        if name in {"real", "fake", "misc"}:
            continue
        src = os.path.join(FRAMES_DIR, name)
        if not os.path.isdir(src):
            continue

        label = label_map.get(name)
        if label is None:
            unmatched.append(name)
            continue

        dst = os.path.join(real_root if label == "real" else fake_root, name)
        safe_merge_move(src, dst)
        if label == "real":
            moved_real += 1
        else:
            moved_fake += 1

    print(f"\n✅ Organized frame folders.")
    print(f"  Moved to real: {moved_real}")
    print(f"  Moved to fake: {moved_fake}")

    if unmatched:
        unmatched_path = os.path.join(FRAMES_DIR, "UNMATCHED_FOLDERS.txt")
        with open(unmatched_path, "w", encoding="utf-8") as f:
            for n in sorted(unmatched):
                f.write(n + "\n")
        print(f"\n⚠️ Unmatched folders: {len(unmatched)}")
        print(f"   Saved list to: {unmatched_path}")
        print("   (These names didn't appear among your *.mp4 files; check if names changed during extraction.)")
    else:
        print("\n🎉 All folders matched and organized.")
