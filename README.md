<img width="1914" height="1008" alt="image" src="https://github.com/user-attachments/assets/766e690f-15f4-409f-9fc0-eadd5085f45a" /># ♻️ EcoVision AI — Waste Segregation & Carbon Footprint Analyser

> **AI-powered computer vision system** that classifies waste into Biodegradable, Recyclable, and Hazardous categories, recommends the correct disposal bin (Indore 6-Bin Model), and estimates carbon footprint reduction potential — all in real-time.

---

## 🎯 Problem Statement

Create a computer vision-based system that:
- **Classifies waste** into biodegradable, recyclable, and hazardous categories
- **Estimates carbon footprint** reduction potential
- Promotes **smart waste management** and supports **circular economy** initiatives
- Provides a **sustainability analytics dashboard**

---

## 🧠 AI Model Details

| Parameter | Value |
|-----------|-------|
| **Base Architecture** | YOLOv8m-seg (Medium) | EfficientNet |
| **Task** | Instance Segmentation |
| **Input Resolution** | 640×640 |
| **Training Epochs** | 40 |

| **Optimizer** | Auto (AdamW) |
| **Learning Rate** | 0.01 → 0.01 (cosine) |
| **Pretrained Backbone** | ✅ Yes (COCO) |
| **AMP (Mixed Precision)** | ✅ Enabled |
| **Augmentations** | Mosaic, Flip, HSV, Scale, RandAugment, Erasing |
| **Device** | GPU (CUDA:0) |





### Waste Classes (6 Categories)

| Class ID | Class Name | Macro Category | Recommended Bin |
|:--------:|-----------|:-------------:|:--------------:|
| 0 | **Plastic** | Recyclable | 🔵 Blue Bin |
| 1 | **Metal** | Recyclable | 🟡 Yellow Bin |
| 2 | **Paper** | Recyclable | 🟢 Green Bin |
| 3 | **Glass** | Recyclable | ⚪ White Bin |
| 4 | **Organic** | Biodegradable | 🟤 Brown Bin |
| 5 | **Hazardous** | Hazardous | 🔴 Red Bin |

---

## 🏗️ Tech Stack — Hybrid AI Pipeline

### AI / Computer Vision Layer
| Component | Technology |
|-----------|-----------|
| Object Detection & Segmentation | **YOLOv8m-seg** (Ultralytics) |
| Framework | **PyTorch** |
| Image Processing | **Pillow + NumPy** |
| Training Data | Custom dataset (6 waste classes) |

### Backend Layer
| Component | Technology |
|-----------|-----------|
| API Server | **Flask** (Python) |
| CORS | **Flask-CORS** |
| Carbon Engine | Custom CO₂/Water/Energy calculator |
| Bin Mapping | **Indore 6-Bin Model** |

### Frontend Layer
| Component | Technology |
|-----------|-----------|
| UI Framework | **React 18** (JSX) |
| Build Tool | **Vite 5** |
| Styling | **TailwindCSS 3** |
| Charts | **Chart.js + react-chartjs-2** |
| Icons | **Lucide React** |
| Routing | **React Router v7** |
| Live Camera | **WebRTC** (`getUserMedia` API) |

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│         📤 Image Upload  OR  📷 Live Camera             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FLASK API SERVER (Port 5000)               │
│                                                         │
│  ┌───────────────────────────────────────────────┐      │
│  │  YOLOv8m-seg Model Inference                  │      │
│  │  • Instance segmentation on 640×640 input     │      │
│  │  • Multi-object detection (6 classes)         │      │
│  │  • Confidence scoring per detection           │      │
│  └──────────────┬────────────────────────────────┘      │
│                 │                                        │
│  ┌──────────────▼────────────────────────────────┐      │
│  │  Post-Processing Pipeline                     │      │
│  │  • Class → Category (Biodeg/Recycl/Hazard)    │      │
│  │  • Indore 6-Bin color-coded assignment         │      │
│  │  • Carbon footprint: CO₂, Water, Energy saved │      │
│  │  • Tree equivalents calculation               │      │
│  │  • Annotated image generation (base64)        │      │
│  └──────────────┬────────────────────────────────┘      │
└─────────────────┼───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              REACT FRONTEND (Port 5173)                 │
│                                                         │
│  • Segmented/annotated image display                    │
│  • Waste category + bin recommendation                  │
│  • Carbon impact dashboard (CO₂, Water, Energy, Trees)  │
│  • Smart disposal suggestions                           │
│  • Real-time sustainability analytics (Charts)          │
│  • Session history and eco score tracking               │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** with pip
- **Node.js 18+** with npm
- GPU recommended (CUDA) for faster inference

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Waste_AI_Project
```

### 2. Install Backend Dependencies
```bash
pip install flask flask-cors ultralytics Pillow numpy
```

### 3. Install Frontend Dependencies
```bash
cd "ecovision-ai - ver2"
npm install
cd ..
```

### 4. Start the Backend Server
```bash
python backend/server.py
```
The Flask API will start on `http://localhost:5000`

### 5. Start the Frontend Dev Server
```bash
cd "ecovision-ai - ver2"
npm run dev
```
The app will open at `http://localhost:5173`

---

## 🔌 API Endpoints

### `POST /api/classify`
Upload a waste image for AI classification.

**Request:** `multipart/form-data` with key `image`

**Response:**
```json
{
  "success": true,
  "primary": {
    "className": "Plastic",
    "mainCategory": "Recyclable",
    "confidence": 87.5,
    "bin": "Blue Bin",
    "binColor": "#3B82F6",
    "carbonImpact": { "co2": 1.8, "water": 10, "energy": 3.5, "treesEquivalent": 0.0857 }
  },
  "detections": [...],
  "annotatedImage": "<base64>",
  "summary": { "totalDetected": 2, "co2Saved": 3.6, "waterSaved": 20, "energySaved": 7 }
}
```

### `GET /api/stats`
Returns cumulative session statistics for the dashboard.

### `GET /api/health`
Health check with loaded model info.

---

## 🌍 Carbon Footprint Factors

| Material | CO₂ Saved (kg/kg) | Water Saved (L/kg) | Energy Saved (kWh/kg) |
|----------|:---------:|:----------:|:-----------:|
| Plastic | 1.8 | 10.0 | 3.5 |
| Metal | 4.0 | 25.0 | 8.0 |
| Paper | 1.1 | 7.0 | 2.5 |
| Glass | 0.6 | 1.5 | 1.2 |
| Organic | 0.5 | 0.5 | 0.3 |
| Hazardous | 2.5 | 15.0 | 5.0 |

---

## 📁 Project Structure

```
Waste_AI_Project/
├── backend/
│   ├── server.py              # Flask API with YOLO inference
│   └── requirements.txt       # Python dependencies
├── ecovision-ai - ver2/       # React Frontend
│   ├── src/
│   │   ├── App.jsx            # Main app layout with navigation
│   │   ├── components/
│   │   │   ├── Landing.jsx        # Hero landing page
│   │   │   ├── Dashboard.jsx      # Sustainability overview
│   │   │   ├── UploadWaste.jsx    # Upload + Live Camera classifier
│   │   │   ├── Analytics.jsx      # Charts and insights
│   │   │   ├── CarbonCalculator.jsx   # Impact calculator
│   │   │   ├── SmartSuggestions.jsx   # Disposal recommendations
│   │   │   └── ui/                # Reusable UI components
│   │   ├── data/dummyData.js  # Fallback data
│   │   └── lib/utils.js       # Utility functions
│   ├── vite.config.js         # Vite config with API proxy
│   ├── tailwind.config.js     # TailwindCSS theme
│   └── package.json
├── waste_seg/                 # YOLO training data & results
│   └── runs/segment/train2/
│       ├── weights/best.pt    # Best trained model weights
│       ├── results.csv        # Training metrics
│       └── confusion_matrix.png
├── app.py                     # Legacy Streamlit app
└── README.md                  # This file
```

---

## 🏆 Key Features

- **🤖 AI-Powered Classification** — YOLOv8m-seg instance segmentation for 6 waste classes
- **📷 Live Camera Vision** — Real-time webcam feed with capture-to-classify
- **🗑️ Indore 6-Bin System** — Color-coded bin recommendations
- **🌱 Carbon Footprint Analysis** — CO₂, water, energy savings per item
- **📊 Sustainability Dashboard** — Real-time charts and eco score tracking
- **💡 Smart Suggestions** — Context-aware disposal recommendations
- **🌓 Dark Mode** — Premium UI with light/dark theme toggle

---

## 👥 Team

Built for promoting smart waste management and supporting circular economy initiatives.

---

## 📄 License

This project is for educational and hackathon purposes.


