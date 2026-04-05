"""
Download sample plant disease images from Plant Village dataset on Kaggle for testing.
This script creates a test folder with images organized by class.

Note: Requires internet connection
Alternative: Use manually collected images or the original training dataset
"""

import os
import urllib.request
from pathlib import Path
import json

# Base directory
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test_images"

# Sample images URLs (using publicly available sources)
# These are example image URLs - please update with valid working URLs
SAMPLE_IMAGES = {
    "Apple___Apple_scab": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Apple___Apple_scab/Apple_scab_10.JPG",
    ],
    "Apple___healthy": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Apple___healthy/healthy_0.JPG",
    ],
    "Tomato___Early_blight": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Tomato___Early_blight/Tomato_Early_blight_10.JPG",
    ],
    "Tomato___healthy": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Tomato___healthy/healthy_10.JPG",
    ],
    "Potato___Late_blight": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Potato___Late_blight/Late_blight_10.JPG",
    ],
    "Potato___healthy": [
        "https://raw.githubusercontent.com/divyat09/plant-disease-detection/master/data/Potato___healthy/healthy_10.JPG",
    ],
}


def create_folder_structure():
    """Create folder structure for test images"""
    class_names = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]
    
    for class_name in class_names:
        class_dir = TEST_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {class_dir}")


def download_images():
    """Download sample images"""
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE IMAGES")
    print("="*60 + "\n")
    
    for class_name, urls in SAMPLE_IMAGES.items():
        class_dir = TEST_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, url in enumerate(urls, 1):
            try:
                filename = f"sample_{idx}.jpg"
                filepath = class_dir / filename
                
                print(f"Downloading {class_name}/{filename}...", end=" ")
                urllib.request.urlretrieve(url, filepath)
                print("✓")
                
            except Exception as e:
                print(f"✗ (Error: {str(e)[:50]})")


def manual_setup_instructions():
    """Print instructions for manual setup"""
    print("\n" + "="*60)
    print("MANUAL SETUP INSTRUCTIONS")
    print("="*60 + "\n")
    
    print("If automatic download fails, you can manually add test images:\n")
    print(f"1. Create folders in: {TEST_DIR}\n")
    print("2. Folders should be named exactly as class names:")
    print("   - Apple___Apple_scab/")
    print("   - Apple___healthy/")
    print("   - Tomato___Early_blight/")
    print("   - Tomato___healthy/")
    print("   - etc.\n")
    print("3. Place image files (.jpg, .png, .jpeg) in each folder\n")
    print("4. Run evaluation:")
    print(f"   python evaluate.py --data \"{TEST_DIR}\"\n")
    
    print("OPTIONS FOR GETTING IMAGES:")
    print("- Plant Village Dataset: https://github.com/spMohanty/PlantVillage-Dataset")
    print("- Kaggle Plant Pathology: https://www.kaggle.com/")
    print("- Your own photos of plant leaves\n")


if __name__ == "__main__":
    print("Setting up test image directory structure...\n")
    
    # Create folder structure
    create_folder_structure()
    
    # Try to download images
    try:
        download_images()
    except Exception as e:
        print(f"\n⚠ Download failed: {e}")
    
    # Print instructions
    manual_setup_instructions()
    
    print(f"Test directory: {TEST_DIR}")
    print("\nTo evaluate, run:")
    print(f"python evaluate.py --data \"{TEST_DIR}\"")
