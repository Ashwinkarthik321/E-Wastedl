

The **E-Waste Scrap Price Predictor** is a Deep Learning–powered application that:

👉 Classifies images of **electronic waste**
👉 Predicts the **scrap price** based on item category
👉 Provides **confidence graph** and **top-3 predictions**
👉 Offers **live camera capture** using Streamlit

This project solves a real-world problem for recycling centers, e-waste shops, and environmental sustainability.

---

# ✨ **Features**

* 📸 **Upload image or use live camera**
* 🧠 **MobileNetV2-based image classifier**
* 💰 **Instant scrap price prediction**
* 📊 **Top-3 prediction probability chart**
* 🎨 **Clean & modern UI (Streamlit)**
* ☁️ **Deployable on Streamlit Cloud**
* ⚡ Fast and lightweight deep learning model

---

# 🧠 **Model Architecture**

* Base Model: **MobileNetV2**
* Training: Transfer Learning + Fine-tuning
* Input Shape: **224 × 224 × 3**
* Accuracy: **90%+**
* Frameworks: **TensorFlow / Keras**

---

# 📂 **Project Structure**

```
E-Wastedl/
│
├── E_waste/
│   ├── App.py                 # Streamlit App
│   ├── predict.py             # CLI Prediction Script
│   ├── banner.jpg             # Homepage Banner
│
├── models/
│   ├── ewaste_classifier.keras   # Trained Model
│   ├── class_indices.json        # Label Mapping
│
├── requirements.txt
├── README.md
```

---

# ⚙️ **Installation**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ashwinkarthik321/E-Wastedl.git
cd E-Wastedl
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ **Running Locally**

Start the Streamlit app:

```bash
streamlit run E_waste/App.py
```

---

# 🌐 **Deployment (Streamlit Cloud)**

1. Go to **[https://share.streamlit.io](https://share.streamlit.io)**
2. Connect your GitHub repo
3. Set the main file path:

```
E_waste/App.py
```

4. Add your requirements.txt
5. Click **Deploy** 🚀
6. [Your live URL will appear!](https://ewastepricemodel.streamlit.app)

---

# 📊 **Scrap Categories**

| Category   | Scrap Price (₹) |
| ---------- | --------------- |
| Battery    | 300             |
| Keyboard   | 150             |
| Microwave  | 2500            |
| Mobile     | 800             |
| Mouse      | 80              |
| PCB        | 200             |
| Player     | 300             |
| Printer    | 500             |
| Television | 1500            |

---

# 🖼 **Screenshots**

### App Home

<img width="1710" height="1073" alt="Screenshot 2025-12-09 at 7 39 23 PM" src="https://github.com/user-attachments/assets/48b79bcb-d8a0-4a6c-aefb-05ba80ebba81" />


### Prediction Example

<img width="1710" height="1073" alt="Screenshot 2025-12-09 at 7 30 18 PM" src="https://github.com/user-attachments/assets/bbf9f73e-a18d-4e5d-959f-91485dd0c45d" />


---

# 🤝 **Contributing**

Contributions are welcome!
You can help by:

* Adding more categories
* Improving UI
* Increasing dataset size
* Enhancing model accuracy

---

# 👨‍💻 **Developer**

**Ashwin**

🌐 GitHub: [https://github.com/Ashwinkarthik321](https://github.com/Ashwinkarthik321)



