# Handwritten Math Symbol & Digit Recognizer

A lightweight, interactive web application built with **Flask** and **PyTorch** that recognizes handwritten mathematical symbols, Greek letters, algebraic variables, and digits in real-time.

Developed as an academic Machine Learning project.

---

## 📌 Project Overview

This project provides an easy-to-use drawing interface where users can hand-draw a math symbol or digit on a canvas. The backend preprocesses the drawing (centering, aspect-ratio scaling, and contrast inversion) and uses a pre-trained **Convolutional Neural Network (CNN)** to predict the symbol, its confidence score, and its corresponding LaTeX code.

### Supported Symbols (47 Classes)
- **Digits (0–9):** `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
- **Arithmetic Operators:** `+`, `-`, `×`, `÷`, `±`, `=`
- **Relational Signs:** `<`, `>`, `≤`, `≥`, `≠`
- **Algebraic Variables:** `a`, `b`, `n`, `x`, `y`, `z`
- **Greek Letters:** `α`, `β`, `γ`, `δ`, `Δ`, `π`, `θ`, `λ`, `σ`
- **Calculus & Analysis:** `√`, `∫`, `∑`, `∏`, `∞`, `∂`
- **Set Theory & Delimiters:** `∈`, `⊂`, `→`, `[`, `]`

---

## 📁 Project Structure

```text
Innovative Assignment/
│
├── static/
│   ├── canvas.js          # Drawing canvas logic and API communication
│   └── style.css          # Responsive styling for the web interface
│
├── templates/
│   └── index.html         # Main user interface
│
├── app.py                 # Flask backend, preprocessing, and model inference
├── model.pth              # Pre-trained CNN weights (2.2 MB, 47 classes)
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation and setup guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed on your system.
- pip package manager.

### 2. Setup Virtual Environment (Recommended)

Open your terminal or command prompt in the project root directory:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Your Web Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 🎨 How to Use the App

1. **Draw:** Use your mouse or touch screen to draw any math symbol or digit inside the canvas.
2. **Stroke Width:** Select *Thin*, *Medium*, or *Thick* brush width depending on your drawing preference.
3. **Predict:** Click **"✨ Predict Symbol"** (or press `Enter` on your keyboard).
4. **View Results:** The model will display:
   - The recognized symbol (Unicode)
   - Symbol name & category
   - Model confidence percentage
   - Formatted LaTeX code with a one-click copy button
   - Top 3 candidate predictions
5. **Clear:** Click **"🧹 Clear"** (or press `Esc` / `C`) to wipe the whiteboard and try another symbol.

---

## 🧠 Model Architecture & Pipeline

1. **Preprocessing Pipeline (`app.py`):**
   - User drawing is received from the canvas as a Base64 PNG.
   - Automatically inverted (white strokes on dark background to match standard handwriting datasets).
   - Noise smoothing via Gaussian filter.
   - Bounding-box detection crops out whitespace and scales the stroke preserving its aspect ratio into a centered 32×32 pixel tensor.

2. **Convolutional Neural Network (`SymbolCNN`):**
   - **Conv Block 1:** Conv2D (1 → 32, 3×3) → BatchNorm → ReLU → MaxPool (2×2)
   - **Conv Block 2:** Conv2D (32 → 64, 3×3) → BatchNorm → ReLU → MaxPool (2×2) → Dropout (0.25)
   - **Classifier:** Linear (4096 → 128) → ReLU → Dropout (0.5) → Linear (128 → 47)
   - Weights stored compactly in `model.pth` (~2.2 MB).

---

## 📜 License & Acknowledgments
- Dataset: Inspired by the HASYv2 handwritten symbol dataset.
- Built for academic Machine Learning coursework.
