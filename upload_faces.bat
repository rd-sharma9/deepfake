@echo off
cd /d "C:\Users\ASUS\OneDrive\Desktop\deepfake-detector"

echo Adding faces dataset to Git...
git lfs track "*.jpg"
git add .gitattributes
git add data/faces/

echo Committing changes...
git commit -m "Add cropped faces dataset"

echo Pushing to GitHub...
git push origin main

echo ✅ Upload complete! Faces dataset is now tracked by Git LFS.
pause
