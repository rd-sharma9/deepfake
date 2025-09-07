@echo off
cd /d "C:\Users\ASUS\OneDrive\Desktop\deepfake-detector"

echo Adding FAKE faces dataset to Git...
git lfs track "*.jpg"
git add .gitattributes
git add data/faces/fake/

echo Committing changes...
git commit -m "Add cropped fake faces dataset"

echo Pushing to GitHub...
git push origin main

echo ✅ Upload complete! FAKE faces dataset is now tracked by Git LFS.
pause
