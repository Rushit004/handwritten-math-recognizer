"""
Handwritten Math Symbol & Digit Recognizer
Flask Web Application for Real-Time Canvas Inference
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import cv2
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# Supported Classes (47 Math Symbols, Digits & Variables)
# ---------------------------------------------------------
CLASSES = {
    0: {"name": "Capital Delta", "symbol": "Δ", "latex": r"\Delta", "category": "Greek"},
    1: {"name": "Alpha", "symbol": "α", "latex": r"\alpha", "category": "Greek"},
    2: {"name": "Beta", "symbol": "β", "latex": r"\beta", "category": "Greek"},
    3: {"name": "Delta", "symbol": "δ", "latex": r"\delta", "category": "Greek"},
    4: {"name": "Digit 0", "symbol": "0", "latex": "0", "category": "Digits"},
    5: {"name": "Digit 1", "symbol": "1", "latex": "1", "category": "Digits"},
    6: {"name": "Digit 2", "symbol": "2", "latex": "2", "category": "Digits"},
    7: {"name": "Digit 3", "symbol": "3", "latex": "3", "category": "Digits"},
    8: {"name": "Digit 4", "symbol": "4", "latex": "4", "category": "Digits"},
    9: {"name": "Digit 5", "symbol": "5", "latex": "5", "category": "Digits"},
    10: {"name": "Digit 6", "symbol": "6", "latex": "6", "category": "Digits"},
    11: {"name": "Digit 7", "symbol": "7", "latex": "7", "category": "Digits"},
    12: {"name": "Digit 8", "symbol": "8", "latex": "8", "category": "Digits"},
    13: {"name": "Digit 9", "symbol": "9", "latex": "9", "category": "Digits"},
    14: {"name": "Division", "symbol": "÷", "latex": r"\div", "category": "Arithmetic"},
    15: {"name": "Equals", "symbol": "=", "latex": "=", "category": "Arithmetic"},
    16: {"name": "Gamma", "symbol": "γ", "latex": r"\gamma", "category": "Greek"},
    17: {"name": "Greater or Equal", "symbol": "≥", "latex": r"\geq", "category": "Relations"},
    18: {"name": "Greater Than", "symbol": ">", "latex": ">", "category": "Relations"},
    19: {"name": "Element Of", "symbol": "∈", "latex": r"\in", "category": "Set / Logic"},
    20: {"name": "Infinity", "symbol": "∞", "latex": r"\infty", "category": "Calculus"},
    21: {"name": "Integral", "symbol": "∫", "latex": r"\int", "category": "Calculus"},
    22: {"name": "Lambda", "symbol": "λ", "latex": r"\lambda", "category": "Greek"},
    23: {"name": "Left Bracket", "symbol": "[", "latex": "[", "category": "Delimiters"},
    24: {"name": "Less or Equal", "symbol": "≤", "latex": r"\leq", "category": "Relations"},
    25: {"name": "Less Than", "symbol": "<", "latex": "<", "category": "Relations"},
    26: {"name": "Minus", "symbol": "-", "latex": "-", "category": "Arithmetic"},
    27: {"name": "Not Equal", "symbol": "≠", "latex": r"\neq", "category": "Relations"},
    28: {"name": "Partial Derivative", "symbol": "∂", "latex": r"\partial", "category": "Calculus"},
    29: {"name": "Pi", "symbol": "π", "latex": r"\pi", "category": "Greek"},
    30: {"name": "Plus", "symbol": "+", "latex": "+", "category": "Arithmetic"},
    31: {"name": "Plus-Minus", "symbol": "±", "latex": r"\pm", "category": "Arithmetic"},
    32: {"name": "Product", "symbol": "∏", "latex": r"\prod", "category": "Calculus"},
    33: {"name": "Right Bracket", "symbol": "]", "latex": "]", "category": "Delimiters"},
    34: {"name": "Right Arrow", "symbol": "→", "latex": r"\rightarrow", "category": "Set / Logic"},
    35: {"name": "Sigma", "symbol": "σ", "latex": r"\sigma", "category": "Greek"},
    36: {"name": "Square Root", "symbol": "√", "latex": r"\sqrt{}", "category": "Calculus"},
    37: {"name": "Subset", "symbol": "⊂", "latex": r"\subset", "category": "Set / Logic"},
    38: {"name": "Summation", "symbol": "∑", "latex": r"\sum", "category": "Calculus"},
    39: {"name": "Theta", "symbol": "θ", "latex": r"\theta", "category": "Greek"},
    40: {"name": "Multiplication", "symbol": "×", "latex": r"\times", "category": "Arithmetic"},
    41: {"name": "Variable a", "symbol": "a", "latex": "a", "category": "Variables"},
    42: {"name": "Variable b", "symbol": "b", "latex": "b", "category": "Variables"},
    43: {"name": "Variable n", "symbol": "n", "latex": "n", "category": "Variables"},
    44: {"name": "Variable x", "symbol": "x", "latex": "x", "category": "Variables"},
    45: {"name": "Variable y", "symbol": "y", "latex": "y", "category": "Variables"},
    46: {"name": "Variable z", "symbol": "z", "latex": "z", "category": "Variables"},
}

# ---------------------------------------------------------
# PyTorch CNN Architecture
# ---------------------------------------------------------
class SymbolCNN(nn.Module):
    """
    Lightweight 2D Convolutional Neural Network for 32x32 handwritten symbols.
    """
    def __init__(self, num_classes=47):
        super(SymbolCNN, self).__init__()
        
        # Convolutional Block 1 (Input: 1x32x32 -> Output: 32x16x16)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 2 (Input: 32x16x16 -> Output: 64x8x8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# Load Pre-Trained Weights
# ---------------------------------------------------------
device = torch.device("cpu")
model = SymbolCNN(num_classes=len(CLASSES))

weights_path = os.path.join(os.path.dirname(__file__), "model.pth")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[OK] Loaded pre-trained model weights from {weights_path}")
else:
    print(f"[WARNING] Model weights file not found at {weights_path}")

# ---------------------------------------------------------
# Image Preprocessing Helper
# ---------------------------------------------------------
def preprocess_image(pil_img, target_size=(32, 32)):
    """
    Preprocess user drawing to match the training format:
    1. Convert to grayscale.
    2. Invert colors so strokes are white on black background.
    3. Find bounding box around the drawn strokes.
    4. Resize proportionally and center inside 32x32 canvas.
    5. Convert to normalized float32 tensor (1, 1, 32, 32).
    """
    # 1. Grayscale conversion
    gray = np.array(pil_img.convert("L"))
    
    # 2. Check polarity (white background with black ink -> invert)
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    if np.mean(corners) > 127:
        inverted = 255 - gray
    else:
        inverted = gray

    # 3. Clean noise with Gaussian blur and binary threshold
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    # 4. Find bounding box of the drawn symbol
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None  # Canvas is blank

    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]

    # 5. Fit inside 24x24 box to leave a comfortable border
    max_dim = max(w, h)
    scale = 22.0 / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. Place centered on 32x32 black background
    canvas = np.zeros(target_size, dtype=np.uint8)
    offset_x = (target_size[1] - new_w) // 2
    offset_y = (target_size[0] - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_crop

    # 7. Normalize pixel values [0, 1] and reshape to (1, 1, 32, 32)
    normalized = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor

# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    """Render the main drawing interface."""
    return render_template("index.html")

@app.route("/classes", methods=["GET"])
def get_classes():
    """Return all supported classes grouped for the UI gallery."""
    return jsonify(CLASSES)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle symbol prediction request from drawing canvas (base64)
    or uploaded file.
    """
    try:
        pil_image = None

        # Check for JSON base64 image data
        if request.is_json:
            data = request.get_json()
            image_data = data.get("image", "")
            if not image_data:
                return jsonify({"success": False, "error": "No image data provided"}), 400
            
            # Remove base64 header if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes))

        # Check for multipart form file upload
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "Empty file uploaded"}), 400
            pil_image = Image.open(file.stream)

        else:
            return jsonify({"success": False, "error": "Invalid request payload"}), 400

        # Preprocess the drawing
        tensor = preprocess_image(pil_image)
        if tensor is None:
            return jsonify({
                "success": False,
                "error": "The canvas appears blank. Please draw a math symbol or digit first!"
            }), 400

        # Run model inference
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]

        # Get top 3 predicted candidates
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_idx = int(top_indices[0])
        top_candidate = CLASSES[top_idx]
        top_confidence = round(float(probabilities[top_idx] * 100), 2)

        # Build list of alternative candidates
        top3_list = []
        for rank, idx in enumerate(top_indices, 1):
            c_info = CLASSES[int(idx)]
            top3_list.append({
                "rank": rank,
                "name": c_info["name"],
                "symbol": c_info["symbol"],
                "latex": c_info["latex"],
                "category": c_info["category"],
                "confidence": round(float(probabilities[idx] * 100), 2)
            })

        return jsonify({
            "success": True,
            "prediction": {
                "name": top_candidate["name"],
                "symbol": top_candidate["symbol"],
                "latex": top_candidate["latex"],
                "category": top_candidate["category"],
                "confidence": top_confidence
            },
            "top3": top3_list
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Math Symbol Recognizer Web Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)
