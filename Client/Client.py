import streamlit as st
import base64
import requests

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Celebrity Recognition", page_icon="🎭", layout="centered")

st.title("🎭 Celebrity Recognition App")
st.write("Upload a photo and let our AI recognize the celebrity!")

# -------------------- Sidebar Info --------------------
st.sidebar.header("⚙️ Settings")

# API URL input (default: local Flask server)
api_url = st.sidebar.text_input(
    "API URL",
    value="http://127.0.0.1:5000/predict",
    help="Enter the backend API endpoint here"
)

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About the App")
st.sidebar.markdown("""
This app predicts which celebrity is in the uploaded photo.  
Currently, only the following **5 celebrities** are supported:

1. 🏏 **Virat Kohli**  
2. 🎾 **Serena Williams**  
3. 🎾 **Roger Federer**  
4. ⚽ **Lionel Messi**  
5. 🎾 **Maria Sharapova**  

⚠️ Please upload a **clear face photo** of one of these celebrities.
""")

# -------------------- Upload Image --------------------
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert uploaded image to base64
    b64_str = base64.b64encode(uploaded_file.read()).decode('utf-8')

    # Call backend API when user clicks button
    if st.button("🔍 Predict Celebrity"):
        with st.spinner("Analyzing the image... Please wait ⏳"):
            try:
                response = requests.post(api_url, json={"image_b64": b64_str})

                if response.status_code == 200:
                    result = response.json().get("prediction", "No result")
                    st.success(f"✨ Prediction: **{result}**")
                else:
                    st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

