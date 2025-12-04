import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import sys
from pathlib import Path

# Resolve paths relative to this script (avoids cwd issues)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "ewaste_classifier.keras"
SAVEDMODEL_PATH = BASE_DIR.parent / "models" / "ewaste_classifier_savedmodel"
CLASS_INDEX_PATH = BASE_DIR.parent / "models" / "class_indices.json"

# Load model
if not MODEL_PATH.exists():
    print(f"Model not found at {MODEL_PATH}")
    sys.exit(1)

model = load_model(str(MODEL_PATH))


# Load class mappings
if not CLASS_INDEX_PATH.exists():
    print(f"Class index file not found at {CLASS_INDEX_PATH}")
    sys.exit(1)

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# Scrap price mapping
scrap_prices = {
    "Battery": 300,
    "Keyboard": 150,
    "Microwave": 2500,
    "Mobile": 800,
    "Mouse": 80,
    "PCB": 200,
    "Player": 300,
    "Printer": 500,
    "Television": 1500
}



def predict_image(img_path):
    img = image.load_img(str(img_path), target_size=(224, 224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, 0)

    pred = model.predict(arr)[0]
    index = np.argmax(pred)
    label = idx_to_class[index]
    confidence = float(pred[index])
    
    price = scrap_prices[label]

    return label, confidence, price

# Test
if __name__ == "__main__":
    test_img = BASE_DIR / "Microwave_67.jpg"
    if not test_img.exists():
        print(f"Test image not found at {test_img}")
        sys.exit(1)

    label, confidence, price = predict_image(test_img)
    print("Predicted Category:", label)
    print("Confidence:", confidence)
    print("Scrap Price:", price)
