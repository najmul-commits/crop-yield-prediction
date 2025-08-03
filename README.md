
# 🌱 Crop Recommendation System

This project is a machine learning-powered web application that recommends the most suitable crop to grow based on soil and environmental conditions. The model is deployed using [Gradio](https://gradio.app/) to provide an interactive interface.

---

## 🚀 Features

- Input parameters:  
  🌾 Nitrogen (N), Phosphorous (P), Potassium (K)  
  🌡 Temperature (°C), 💧 Humidity (%), pH, Rainfall (mm)

- Model predicts the **best crop** based on inputs.

- Clean and user-friendly UI built using **Gradio**.

---

## 📁 Files

- `crops_perdictionFinal (1).ipynb` – Jupyter notebook used for training the machine learning model.
- `crops.joblib` – Serialized trained model using `joblib`.
- `gradio_app.py` – Python script with Gradio interface for crop prediction.
- `requirements.txt` – Python dependencies for the project.

---

## 🧪 Installation

1. **Clone the repo or download files**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Gradio app**

```bash
python gradio_app.py
```

4. Open the provided Gradio URL in your browser.

---

## 📦 Dependencies

- `gradio` – For creating the web interface
- `scikit-learn` – For model development
- `pandas` – For data manipulation
- `joblib` – For saving/loading the ML model

---

## 🧠 Model Info

- Supervised machine learning model (e.g., RandomForestClassifier) trained on a crop recommendation dataset.
- Input features: N, P, K, temperature, humidity, pH, rainfall.
- Output: Recommended crop name.

---

## ✅ Example Input

| N  | P  | K  | Temp (°C) | Humidity (%) | pH  | Rainfall (mm) |
|----|----|----|------------|----------------|-----|-----------------|
| 90 | 42 | 43 | 20.8       | 82.0           | 6.5 | 200             |

→ Output: 🌾 Recommended Crop: **rice**

---

## 📌 Notes

- Make sure `crops.joblib` is in the same directory as `gradio_app.py`.
- Tested with Python 3.7+

---

## 📜 License

This project is for educational and research purposes.
