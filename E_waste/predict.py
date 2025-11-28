import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import pandas as pd

# Load model
try:
    model = load_model('models/ewaste_classifier.keras')
except Exception as e:
    st.error("❌ Model file is corrupted or incompatible.")
    st.exception(e)
    st.stop()


# Load class mappings
with open('models/class_indices.json') as f:
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
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, 0)

    pred = model.predict(arr)[0]
    index = np.argmax(pred)
    label = idx_to_class[index]
    confidence = float(pred[index])
    
    price = scrap_prices[label]

    return label, confidence, price

# Test
# print(predict_image(" /Microwave_67.jpg"))
if __name__ == "__main__":
    label, confidence, price = predict_image("/workspaces/springboot1/E_waste/Microwave_67.jpg")
    print("Predicted Category:", label)
    print("Confidence:", confidence)
    print("Scrap Price:", price)
