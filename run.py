import os
import subprocess

# ✅ Path to your dataset
DATA_DIR = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\faces"

# ✅ Define remote repo (set this first)
REMOTE_REPO = "origin"   # change if you use another remote
BRANCH = "main"          # or "master" depending on your repo

def git_push(folder_name):
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} does not exist")
        return

    print(f"📂 Adding {folder_name} to Git...")
    subprocess.run(["git", "lfs", "track", "*.jpg"], check=True)
    subprocess.run(["git", "add", folder_path], check=True)
    subprocess.run(["git", "commit", "-m", f"Add {folder_name} faces dataset"], check=True)
    subprocess.run(["git", "push", REMOTE_REPO, BRANCH], check=True)
    print(f"✅ Pushed {folder_name} successfully!")

if __name__ == "__main__":
    # Push real faces
    git_push("real")

    # Push fake faces
    git_push("fake")
