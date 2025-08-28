import os
import shutil

def remove_unnecessary_000(base_dir):
    """
    Removes inner '000' folders if present inside any video folder.
    Example: real/video1/000/ will be deleted.
    """
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d == "000":  # look for '000' folder
                folder_to_delete = os.path.join(root, d)
                print(f"[DELETE] Removing unnecessary folder: {folder_to_delete}")
                shutil.rmtree(folder_to_delete)

if __name__ == "__main__":
    base_data = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\frames"

    real_path = os.path.join(base_data, "real")
    fake_path = os.path.join(base_data, "fake")

    print("[START] Cleaning REAL dataset")
    remove_unnecessary_000(real_path)

    print("[START] Cleaning FAKE dataset")
    remove_unnecessary_000(fake_path)

    print("[DONE] Unnecessary '000' folders removed ✅")
