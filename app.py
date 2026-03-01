import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the ORIGINAL best model (train2 was our best at mAP50=36.3%)
model = YOLO("waste_seg/runs/segment/train2/weights/best.pt")

# Bin mapping
BIN_MAPPING = {
    "Plastic": "Blue Bin",
    "Metal": "Yellow Bin",
    "Paper": "Green Bin",
    "Glass": "White Bin",
    "Organic": "Brown Bin",
    "Hazardous": "Red Bin"
}

# Carbon impact levels
CARBON_IMPACT = {
    "Plastic": "High",
    "Metal": "Medium",
    "Paper": "Low",
    "Glass": "Low",
    "Organic": "Very Low",
    "Hazardous": "Very High"
}

st.set_page_config(page_title="AI Waste Segregation", layout="wide")

st.title("♻️ AI Waste Segregation & Carbon Analyzer")

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing Waste..."):
        results = model.predict(image, conf=0.25)

    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Segmentation Result", use_column_width=True)

    st.subheader("Detected Waste Details")

    detected = []

    for cls in results[0].boxes.cls:
        class_name = model.names[int(cls)]
        detected.append(class_name)

    if len(detected) > 0:
        for waste in set(detected):
            st.markdown(f"### 🗑 Waste Type: {waste}")
            st.write(f"Recommended Bin: **{BIN_MAPPING.get(waste, 'Unknown')}**")
            st.write(f"Carbon Impact Level: **{CARBON_IMPACT.get(waste, 'Unknown')}**")
            st.write("---")
    else:
        st.warning("No waste detected.")