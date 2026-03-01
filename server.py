"""
EcoVision Backend – Flask API for AI Waste Classification
Loads a trained YOLOv11-seg model and exposes REST endpoints
for the React frontend.
"""

import io
import base64
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "waste_seg" / "runs" / "segment" / "train2" / "weights" / "best.pt")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load YOLO model once at startup
# ---------------------------------------------------------------------------
from ultralytics import YOLO

print(f"[EcoVision] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"[EcoVision] Model loaded. Classes: {model.names}")

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

# Map YOLO class-name → macro category (Biodegradable / Recyclable / Hazardous)
CLASS_TO_CATEGORY = {
    # Common TACO / waste dataset classes → category
    "Plastic":      "Recyclable",
    "Metal":        "Recyclable",
    "Paper":        "Recyclable",
    "Cardboard":    "Recyclable",
    "Glass":        "Recyclable",
    "Organic":      "Biodegradable",
    "Food":         "Biodegradable",
    "Vegetation":   "Biodegradable",
    "Hazardous":    "Hazardous",
    "Battery":      "Hazardous",
    "E-waste":      "Hazardous",
    "Chemical":     "Hazardous",
    "Textile":      "Recyclable",
    "Wood":         "Biodegradable",
    "Rubber":       "Hazardous",
    "Leather":      "Recyclable",
    "Ceramic":      "Recyclable",
}

# Indore 6-Bin system
CATEGORY_TO_BIN = {
    "Plastic":    {"bin": "Blue Bin",   "color": "#3B82F6"},
    "Metal":      {"bin": "Yellow Bin", "color": "#EAB308"},
    "Paper":      {"bin": "Green Bin",  "color": "#22C55E"},
    "Cardboard":  {"bin": "Green Bin",  "color": "#22C55E"},
    "Glass":      {"bin": "White Bin",  "color": "#E5E7EB"},
    "Organic":    {"bin": "Brown Bin",  "color": "#92400E"},
    "Food":       {"bin": "Brown Bin",  "color": "#92400E"},
    "Vegetation": {"bin": "Brown Bin",  "color": "#92400E"},
    "Hazardous":  {"bin": "Red Bin",    "color": "#EF4444"},
    "Battery":    {"bin": "Red Bin",    "color": "#EF4444"},
    "E-waste":    {"bin": "Red Bin",    "color": "#EF4444"},
    "Chemical":   {"bin": "Red Bin",    "color": "#EF4444"},
    "Textile":    {"bin": "Blue Bin",   "color": "#3B82F6"},
    "Wood":       {"bin": "Brown Bin",  "color": "#92400E"},
    "Rubber":     {"bin": "Red Bin",    "color": "#EF4444"},
    "Leather":    {"bin": "Blue Bin",   "color": "#3B82F6"},
    "Ceramic":    {"bin": "White Bin",  "color": "#E5E7EB"},
}

# Carbon-footprint reduction potential (kg CO₂ saved per kg of material)
CARBON_FACTORS = {
    "Plastic":    {"co2": 1.8, "water": 10,  "energy": 3.5},
    "Metal":      {"co2": 4.0, "water": 25,  "energy": 8.0},
    "Paper":      {"co2": 1.1, "water": 7,   "energy": 2.5},
    "Cardboard":  {"co2": 1.1, "water": 7,   "energy": 2.5},
    "Glass":      {"co2": 0.6, "water": 1.5, "energy": 1.2},
    "Organic":    {"co2": 0.5, "water": 0.5, "energy": 0.3},
    "Food":       {"co2": 0.5, "water": 0.5, "energy": 0.3},
    "Vegetation": {"co2": 0.3, "water": 0.3, "energy": 0.1},
    "Hazardous":  {"co2": 2.5, "water": 15,  "energy": 5.0},
    "Battery":    {"co2": 3.0, "water": 20,  "energy": 6.0},
    "E-waste":    {"co2": 5.0, "water": 30,  "energy": 10.0},
    "Chemical":   {"co2": 2.0, "water": 12,  "energy": 4.0},
    "Textile":    {"co2": 1.5, "water": 8,   "energy": 3.0},
    "Wood":       {"co2": 0.4, "water": 1.0, "energy": 0.5},
    "Rubber":     {"co2": 2.0, "water": 10,  "energy": 4.0},
    "Leather":    {"co2": 1.5, "water": 8,   "energy": 3.0},
    "Ceramic":    {"co2": 0.5, "water": 1.0, "energy": 1.0},
}

# Default fallback
DEFAULT_CARBON = {"co2": 1.0, "water": 5.0, "energy": 2.0}
CO2_PER_TREE_YEAR = 21  # kg CO₂ absorbed by one tree per year

def get_category(class_name):
    """Return macro category for a YOLO class name."""
    for key, cat in CLASS_TO_CATEGORY.items():
        if key.lower() in class_name.lower():
            return cat
    # Heuristic fallback
    return "Recyclable"

def get_bin(class_name):
    """Return bin info for a class."""
    for key, info in CATEGORY_TO_BIN.items():
        if key.lower() in class_name.lower():
            return info
    return {"bin": "Blue Bin", "color": "#3B82F6"}

def get_carbon(class_name):
    """Return carbon-footprint factors for a class."""
    for key, factors in CARBON_FACTORS.items():
        if key.lower() in class_name.lower():
            return factors
    return DEFAULT_CARBON

# ---------------------------------------------------------------------------
# Session stats (in-memory for demo)
# ---------------------------------------------------------------------------
session_stats = {
    "totalScans": 0,
    "totalDetections": 0,
    "categoryBreakdown": {"Biodegradable": 0, "Recyclable": 0, "Hazardous": 0},
    "totalCo2Saved": 0.0,
    "totalWaterSaved": 0.0,
    "totalEnergySaved": 0.0,
    "recentActivity": [],  # last 10 scans
}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/classify", methods=["POST"])
def classify():
    """Accept an image upload, run YOLO inference, return structured results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as form-data with key 'image'."}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Run inference
    results = model.predict(image, conf=0.25)
    result = results[0]

    # --- Build detections list ---
    detections = []
    category_counts = {"Biodegradable": 0, "Recyclable": 0, "Hazardous": 0}
    total_co2 = 0.0
    total_water = 0.0
    total_energy = 0.0

    if result.boxes is not None and len(result.boxes):
        for i, cls_tensor in enumerate(result.boxes.cls):
            class_name = model.names[int(cls_tensor)]
            conf = float(result.boxes.conf[i]) * 100  # percentage
            category = get_category(class_name)
            bin_info = get_bin(class_name)
            carbon = get_carbon(class_name)

            detection = {
                "className": class_name,
                "mainCategory": category,
                "confidence": round(conf, 1),
                "bin": bin_info["bin"],
                "binColor": bin_info["color"],
                "carbonImpact": {
                    "co2": carbon["co2"],
                    "water": carbon["water"],
                    "energy": carbon["energy"],
                    "treesEquivalent": round(carbon["co2"] / CO2_PER_TREE_YEAR, 4),
                }
            }
            detections.append(detection)
            category_counts[category] = category_counts.get(category, 0) + 1
            total_co2 += carbon["co2"]
            total_water += carbon["water"]
            total_energy += carbon["energy"]

    # --- Annotated image as base64 ---
    annotated_array = result.plot()  # numpy BGR
    annotated_rgb = annotated_array[:, :, ::-1]  # convert BGR → RGB
    pil_annotated = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_annotated.save(buf, format="JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- Determine primary result (highest confidence) ---
    primary = None
    if detections:
        primary = max(detections, key=lambda d: d["confidence"])

    # --- Update session stats ---
    session_stats["totalScans"] += 1
    session_stats["totalDetections"] += len(detections)
    for cat, count in category_counts.items():
        session_stats["categoryBreakdown"][cat] = session_stats["categoryBreakdown"].get(cat, 0) + count
    session_stats["totalCo2Saved"] += total_co2
    session_stats["totalWaterSaved"] += total_water
    session_stats["totalEnergySaved"] += total_energy

    # Recent activity log
    activity_entry = {
        "scanNumber": session_stats["totalScans"],
        "detections": len(detections),
        "primaryCategory": primary["mainCategory"] if primary else "None",
        "primaryClass": primary["className"] if primary else "None",
        "confidence": primary["confidence"] if primary else 0,
    }
    session_stats["recentActivity"].insert(0, activity_entry)
    session_stats["recentActivity"] = session_stats["recentActivity"][:10]

    # Build response
    response = {
        "success": True,
        "detections": detections,
        "annotatedImage": annotated_b64,
        "primary": primary,
        "summary": {
            "totalDetected": len(detections),
            "categoryBreakdown": category_counts,
            "co2Saved": round(total_co2, 2),
            "waterSaved": round(total_water, 2),
            "energySaved": round(total_energy, 2),
            "treesEquivalent": round(total_co2 / CO2_PER_TREE_YEAR, 4),
        }
    }

    return jsonify(response)


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return cumulative session statistics for the dashboard."""
    total = sum(session_stats["categoryBreakdown"].values()) or 1
    breakdown = [
        {"category": "Biodegradable", "value": round(session_stats["categoryBreakdown"]["Biodegradable"] / total * 100)},
        {"category": "Recyclable",    "value": round(session_stats["categoryBreakdown"]["Recyclable"] / total * 100)},
        {"category": "Hazardous",     "value": round(session_stats["categoryBreakdown"]["Hazardous"] / total * 100)},
    ]

    # Calculate eco score (0-100 based on recycling ratio)
    recyclable_pct = session_stats["categoryBreakdown"]["Recyclable"] / total * 100
    bio_pct = session_stats["categoryBreakdown"]["Biodegradable"] / total * 100
    eco_score = min(100, round(recyclable_pct * 0.6 + bio_pct * 0.4 + session_stats["totalScans"] * 2))

    return jsonify({
        "totalScans": session_stats["totalScans"],
        "totalDetections": session_stats["totalDetections"],
        "totalCo2Saved": round(session_stats["totalCo2Saved"], 2),
        "totalWaterSaved": round(session_stats["totalWaterSaved"], 2),
        "totalEnergySaved": round(session_stats["totalEnergySaved"], 2),
        "ecoScore": eco_score,
        "wasteBreakdown": breakdown,
        "recentActivity": session_stats["recentActivity"],
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "classes": model.names})


if __name__ == "__main__":
    print("[EcoVision] Starting Flask API server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
