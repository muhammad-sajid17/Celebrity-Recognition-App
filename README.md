# Celebrity Face Recognition Project 🎭

This project is an end-to-end **Machine Learning + Web App** system that recognizes faces of 5 celebrities.  
It includes a **Flask backend API** for model inference and a **Streamlit frontend** where users can upload an image to test predictions.

---

## 🚀 Project Structure
```
ML-Project/
│── Celebrities/         # Raw + Cropped celebrity images (used for training)
│── Client/              # Streamlit frontend
│   └── requirements.txt
│── Server/              # Flask backend + model files
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── deployment.py
│   └── requirements.txt
│── Test Data/           # Test images for evaluation
│── Jupyter Notebook/    # Final training notebook
│── README.md
│── requirements.txt     # Combined requirements (for full setup)
│── .gitignore
```

---

## ⚡ Features
- Face detection & cropping using **OpenCV**  
- Feature extraction with **Wavelet Transform**  
- Hyper Parameter Tuning with **GridSearchCV**  
- Classification using a **trained ML model** (`scikit-learn`)  
- Flask API for serving predictions
- Streamlit frontend for uploading an image and seeing results 

---

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/celebrity-face-recognition.git
cd celebrity-face-recognition
```

### Install Dependencies
For the **whole project**:
```bash
pip install -r requirements.txt
```

Or separately:
```bash
pip install -r Client/requirements.txt
pip install -r Server/requirements.txt
```

---

## ▶️ Running the Project

### 1️⃣ Start the Backend (Flask API)
```bash
cd Server
python deployment.py
```
By default, it runs on `http://127.0.0.1:5000`.

### 2️⃣ Start the Frontend (Streamlit App)
```bash
cd Client
streamlit run app.py
```

The Streamlit app will open in your browser.

---

## 🌐 Deployment

- **Backend (Flask API):** Deploy on [Render](https://render.com) or [Railway](https://railway.app).  
- **Frontend (Streamlit):** Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud).  

After deployment:
- Update the API URL in `Client/app.py` so Streamlit calls the deployed API.

---

## 🎯 Accepted Celebrities
Currently, the model supports **5 celebrities only**:
- Virat Kohli 
- Serena Williams  
- Roger Federed  
- Lionel Messi  
- Maria Sharapova 


---

## 📊 Model Training
The model was trained in the `Jupyter Notebook/celebrity_training.ipynb`.  
It uses:
- **Face detection** with OpenCV Haarcascade  
- **Wavelet transforms** for feature extraction  
- **Logistic Regression / SVM / RandomForest** (choose best based on accuracy)

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
This project is licensed under the MIT License.
