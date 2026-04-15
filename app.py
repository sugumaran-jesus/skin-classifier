from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# -----------------------------
# MODEL CONFIG
# -----------------------------
MODEL_PATH = "model/skin_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1bqwEkTMcJODBNfsUpAsdDDM3PFYaszYY"

os.makedirs("model", exist_ok=True)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download complete")
    except Exception as e:
        print("Download failed:", e)

# -----------------------------
# DOWNLOAD MODEL (SAFE)
# -----------------------------
os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        print("Model download failed:", e)

# -----------------------------
# LAZY MODEL LOADING (IMPORTANT)
# -----------------------------
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# -----------------------------
# CLASSES
# -----------------------------
classes = ["acne", "dry", "normal", "oily"]

# -----------------------------
# ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        result = None
        suggestion = None

        if request.method == "POST":

            if "image" not in request.files:
                return "No image uploaded"

            file = request.files["image"]

            if file.filename == "":
                return "No file selected"

            # IMAGE PREPROCESSING
            img = Image.open(file).convert("RGB").resize((128, 128))
            img = np.array(img) / 255.0
            img = img.reshape(1, 128, 128, 3)

            # LOAD MODEL SAFELY
            mdl = load_model()

            # PREDICTION
            prediction = mdl.predict(img)
            result = classes[np.argmax(prediction)]

            # -----------------------------
            # LONG SUGGESTIONS
            # -----------------------------
            if result == "dry":
                suggestion = """
Your skin is dry. It lacks natural oils and moisture, which can cause tightness, flakiness, and irritation.

Recommended care:
- Use a gentle hydrating cleanser twice daily
- Apply a rich moisturizer immediately after washing
- Drink plenty of water throughout the day
- Avoid hot water and harsh soaps
- Use products with hyaluronic acid or ceramides
"""

            elif result == "oily":
                suggestion = """
Your skin is oily, meaning it produces excess sebum that can lead to shine and clogged pores.

Recommended care:
- Wash your face twice daily with oil-free cleanser
- Use non-comedogenic skincare products
- Avoid heavy creams and greasy foods
- Use gel-based lightweight moisturizers
- Consider salicylic acid for oil control
"""

            elif result == "acne":
                suggestion = """
Your skin shows signs of acne caused by clogged pores, bacteria, or excess oil production.

Recommended care:
- Clean your face gently twice daily
- Avoid touching or squeezing pimples
- Use products with salicylic acid or benzoyl peroxide
- Keep pillow covers and towels clean
- Maintain a healthy diet and hydration
"""

            else:
                suggestion = """
Your skin is normal and well-balanced.

Recommended care:
- Maintain a simple skincare routine
- Cleanse and moisturize daily
- Use sunscreen for protection
- Stay hydrated and eat healthy
- Avoid overusing skincare products
"""

        return render_template("index.html", result=result, suggestion=suggestion)

    except Exception as e:
        return f"Server Error: {str(e)}"

# -----------------------------
# RUN APP (LOCAL ONLY)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)