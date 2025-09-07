# Make sure we’re inside the repo
cd "C:\Users\ASUS\OneDrive\Desktop\deepfake-detector"

# Reset staged changes (optional, just to clean state)
git reset

# Add ONLY fake faces
git add data/faces/fake

# Commit them
git commit -m "Add fake faces dataset only"

# Push to GitHub
git push origin main
