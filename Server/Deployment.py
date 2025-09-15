import cv2
import numpy as np
import urllib.request
import pickle
import joblib
import CroppedFaces as cf
import WaveletTransformation as wt  # your w2d

# Load scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

svcmodel = joblib.load('Celebrity Image.pkl')

# Class dictionary
celebrities = {
    0: 'Lionel Messi',
    1: 'Maria Sharapove',
    2: 'Roger Federer',
    3: 'Serena Williams',
    4: 'Virat Kohli'
}

# ---------- Helper to read image ----------
# def load_image(image_path):
#     if image_path.startswith("http://") or image_path.startswith("https://"):
#         resp = urllib.request.urlopen(image_path)
#         image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
#         img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
#     else:
#         img = cv2.imread(image_path)
#     return img

# ---------- Feature Extraction ----------
def features(image_path):
    # raw_img = load_image(image_path)
    # raw_img = cv2.imread(image_path)
    if image_path is None:
        return None
    
    image = cf.get_cropped_face(image_path)
    if image is None:
        return None

    scaled_raw_image = cv2.resize(image, (32, 32))
    har_image = wt.w2d(image, 'db1', 5)
    scaled_har_image = cv2.resize(har_image, (32, 32))

    raw_flat = scaled_raw_image.reshape(-1, 1)
    har_flat = scaled_har_image.reshape(-1, 1)

    combined_image = np.vstack((raw_flat, har_flat))
    return combined_image.flatten()

# ---------- Wrapper ----------
def real_features(image_path):
    feature = features(image_path)
    if feature is None:
        return None
    feature = np.array(feature).reshape(1, -1)
    feature = scaler.transform(feature)
    return feature

# def predict_celebrity(image_path):
#     feats = real_features(image_path)
#     if feats is None:
#         return "No face detected in the image."
#     pred = svcmodel.predict(feats)[0]
#     return celebrities[pred]

import base64

def load_b64_image(b64_string):
    # Decode base64 string to bytes
    image_data = base64.b64decode(b64_string)
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode numpy array to OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def predict_celebrity(b64_string):
    img = load_b64_image(b64_string)
    feats = real_features(img)
    if feats is None:
        return "No face detected in the image."
    pred = svcmodel.predict(feats)[0]
    return celebrities[pred]


# prediction = predict_celebrity('../Test Data/1.jpeg')
# print(prediction)
with open("../Test Data/1.jpeg", "rb") as f:
    b64_str = base64.b64encode(f.read()).decode('utf-8')

prediction = predict_celebrity(b64_str)
print(prediction)
