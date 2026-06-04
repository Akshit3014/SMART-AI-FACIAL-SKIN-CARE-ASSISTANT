# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- PATHS ----------------
MODEL_PATH = "skin_type_model.h5"
CSV_FILE = "skin_analysis_history.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Could not load model. Train it first using train_model.py.")
    st.stop()

# ---------------- SKIN CARE DATA ----------------
skin_data = {
    ("Oily", "Summer"): {
        "Oiliness": "80–90%",
        "Dryness": "10–20%",
        "Water": "2.5–3.0 L/day",
        "SPF": "SPF 40–50 (gel-based, non-comedogenic)",
        "Moisturizer": "3/5",
        "Type": "Light gel / water-based",
        "Tips": "Wash face 2–3 times daily, avoid heavy creams"
    },
    ("Oily", "Winter"): {
        "Oiliness": "70–80%",
        "Dryness": "20–30%",
        "Water": "2.5–3.0 L/day",
        "SPF": "SPF 30–40",
        "Moisturizer": "4/5",
        "Type": "Lightweight lotion with hyaluronic acid",
        "Tips": "Use mild cleanser, hydrate skin daily"
    },
    ("Dry", "Summer"): {
        "Oiliness": "20–30%",
        "Dryness": "70–80%",
        "Water": "3.0–3.5 L/day",
        "SPF": "SPF 15–30 (cream-based)",
        "Moisturizer": "4/5",
        "Type": "Hydrating cream with aloe vera",
        "Tips": "Avoid hot showers, apply moisturizer twice daily"
    },
    ("Dry", "Winter"): {
        "Oiliness": "10–20%",
        "Dryness": "80–90%",
        "Water": "3.5–4.0 L/day",
        "SPF": "SPF 20–30",
        "Moisturizer": "5/5",
        "Type": "Deep hydrating (shea butter, ceramides)",
        "Tips": "Use humidifier indoors, avoid alcohol-based toner"
    },
    ("Normal", "Summer"): {
        "Oiliness": "40–50%",
        "Dryness": "40–50%",
        "Water": "2.5–3.0 L/day",
        "SPF": "SPF 30–50",
        "Moisturizer": "3/5",
        "Type": "Light lotion or gel",
        "Tips": "Maintain balanced diet, use gentle cleanser"
    },
    ("Normal", "Winter"): {
        "Oiliness": "30–40%",
        "Dryness": "50–60%",
        "Water": "2.5–3.5 L/day",
        "SPF": "SPF 20–30",
        "Moisturizer": "4/5",
        "Type": "Hydrating cream",
        "Tips": "Avoid hot showers, moisturize daily"
    },
}

# ---------------- FUNCTIONS ----------------
labels = ["Dry", "Normal", "Oily"]

skin_map = {
    "Dry": {"Oiliness": 15, "Dryness": 85},
    "Normal": {"Oiliness": 50, "Dryness": 50},
    "Oily": {"Oiliness": 85, "Dryness": 15}
}

def predict_skin_type(img_path):
    """Predict skin type and return class, confidence, oiliness %, dryness %."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(img_arr)[0]  # probabilities
    
    # Predicted class & confidence
    predicted_class = labels[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)
    
    # Weighted Oiliness & Dryness %
    oiliness = sum(preds[i] * skin_map[labels[i]]["Oiliness"] for i in range(3))
    dryness  = sum(preds[i] * skin_map[labels[i]]["Dryness"] for i in range(3))
    
    return predicted_class, confidence, round(oiliness, 1), round(dryness, 1)

def get_recommendation(season):
    """Return recommendations based on season (using Oily as default)."""
    return skin_data[("Oily", season)]

def create_pdf_report(user_name, pred_class, rec, oiliness, dryness):
    pdf_path = os.path.join(OUTPUT_DIR, f"{user_name}_Skin_Report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-60, "AI Smart Skin Care Report")
    c.line(40, height-70, width-40, height-70)
    
    # Details
    c.setFont("Helvetica", 12)
    y = height - 110
    line_space = 20
    info_lines = [
        f"User Name: {user_name}",
        f"Predicted Skin Type: {pred_class}",
        f"Oiliness: {oiliness}%    Dryness: {dryness}%",
        f"Recommended Water Intake: {rec['Water']}",
        f"SPF Recommendation: {rec['SPF']}",
        f"Moisturizer: {rec['Moisturizer']} ({rec['Type']})",
        "",
        "Tips:",
        f"{rec['Tips']}"
    ]
    for line in info_lines:
        c.drawString(60, y, line)
        y -= line_space
        if y < 200:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 100
    c.save()
    return pdf_path

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="AI Smart Skin Care", layout="wide")
st.title("AI Smart Skin Care Assistant")

menu = ["Analyze Skin", "View Past Reports", "About Project"]
choice = st.sidebar.radio("Navigation", menu)

# ================ ANALYZE SKIN ================
if choice == "Analyze Skin":
    st.subheader("Upload or Capture Your Image")
    
    user_name = st.text_input("Enter Your Name", "User")
    source = st.radio("Choose Image Source", ["Capture Real-time", "Upload File"])
    
    captured_img = None
    if source == "Capture Real-time":
        captured_img = st.camera_input("Capture Image")
    else:
        captured_img = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])
    
    season = st.selectbox("Select Current Season", ["Summer", "Winter"])
    
    if captured_img is not None:
        img_path = os.path.join(OUTPUT_DIR, "temp_image.jpg")
        with open(img_path, "wb") as f:
            f.write(captured_img.getbuffer())
        
        st.image(img_path, caption="Your Selected Image", width=300)
        
        if st.button("Analyze Skin"):
            st.info("Analyzing skin type...")
            
            # Step 1: Predict
            pred_class, confidence, oiliness, dryness = predict_skin_type(img_path)
            
            # Step 2: Get recommendations based on season
            rec = get_recommendation(season)
            
            # Step 3: Save history
            new_entry = {
                "Name": user_name,
                "PredictedClass": pred_class,
                "Confidence": f"{confidence:.2f}%",
                "Oiliness": f"{oiliness}%",
                "Dryness": f"{dryness}%",
                "Season": season,
                "Water": rec["Water"],
                "SPF": rec["SPF"],
                "Moisturizer": rec["Moisturizer"],
                "Type": rec["Type"],
                "Tips": rec["Tips"],
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            df_new = pd.DataFrame([new_entry])
            if os.path.exists(CSV_FILE):
                df_old = pd.read_csv(CSV_FILE)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(CSV_FILE, index=False)
            
            # Step 4: PDF Report
            pdf_path = create_pdf_report(user_name, pred_class, rec, oiliness, dryness)
            
            # Step 5: Display results
            st.markdown(f"### Skin Analysis Result for {user_name}")
            st.write(f"Predicted Skin Type: **{pred_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.write(f"Oiliness: **{oiliness}%**  Dryness: **{dryness}%**")
            st.write(f"Recommendations based on calculated Oiliness & Dryness:")
            st.write(f"- Water Intake: {rec['Water']}")
            st.write(f"- SPF: {rec['SPF']}")
            st.write(f"- Moisturizer: {rec['Moisturizer']} ({rec['Type']})")
            st.write(f"- Tips: {rec['Tips']}")
            
            col1, col2 = st.columns(2)
            with col1:
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name=os.path.basename(pdf_path))
            with col2:
                with open(img_path, "rb") as f:
                    st.download_button("Download Image", f, file_name=os.path.basename(img_path))

# ================ VIEW HISTORY ================
elif choice == "View Past Reports":
    st.subheader("Your Saved Skin Analysis History")
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            st.dataframe(df)
        except Exception:
            st.warning("History file is corrupted. Delete it and try again.")
    else:
        st.info("No reports found yet. Please analyze your skin first.")

# ================ ABOUT ================
else:
    st.subheader("About Project")
    st.write("""
AI Smart Skin Care is an ML-powered application that:
- Predicts your skin type (Oily, Dry, Normal)
- Converts CNN confidence → Oiliness % & Dryness %
- Provides personalized recommendations & tips based on season
- Generates a PDF report and saves history
Built using TensorFlow and Streamlit.
""")
