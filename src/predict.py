"""
Inference Pipeline Module
Handwritten Symbol Classifier

Provides end-to-end inference from an input image:
- Preprocessing (inversion, thresholding, centering, normalization)
- Feature extraction (Raw Pixels vs HOG vs 2D Tensors)
- Prediction with confidence and Top-K ranked candidates
- Explainability outputs (intermediate preprocessing stages & HOG gradient visualization)
"""

import os
import json
import joblib
import numpy as np
import torch
from PIL import Image

from src.data_preprocessing import preprocess_single_image
from src.feature_engineering import FeaturePipeline, get_hog_visualization
from src.train import SymbolCNN


class SymbolPredictor:
    """
    End-to-end inference engine supporting multiple trained models
    and explainability breakdowns.
    """
    def __init__(
        self,
        metadata_path="data/processed/metadata.json",
        models_dir="models"
    ):
        self.models_dir = models_dir
        self.metadata_path = metadata_path

        # Load metadata
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.num_classes = self.metadata["num_classes"]
        self.class_names = [self.metadata["idx_to_class"][str(i)] for i in range(self.num_classes)]
        self.class_stats = self.metadata["class_statistics"]

        # Load feature pipeline
        pipeline_path = os.path.join(models_dir, "feature_pipeline.joblib")
        if os.path.exists(pipeline_path):
            self.feat_pipe = FeaturePipeline.load(pipeline_path)
        else:
            self.feat_pipe = None

        # Load models
        self.loaded_models = {}
        self._load_available_models()

    def _load_available_models(self):
        # 1. Baseline LogReg Raw
        p = os.path.join(self.models_dir, "baseline_logreg_raw.joblib")
        if os.path.exists(p):
            self.loaded_models["Baseline (LogReg Raw)"] = {"model": joblib.load(p), "type": "sklearn", "feat_type": "raw"}

        # 2. Baseline LogReg HOG
        p = os.path.join(self.models_dir, "baseline_logreg_hog.joblib")
        if os.path.exists(p):
            self.loaded_models["Baseline (LogReg HOG)"] = {"model": joblib.load(p), "type": "sklearn", "feat_type": "hog"}

        # 3. SVM (HOG)
        p = os.path.join(self.models_dir, "best_svm.joblib")
        if os.path.exists(p):
            self.loaded_models["SVM (HOG)"] = {"model": joblib.load(p), "type": "sklearn", "feat_type": "hog"}

        # 4. Random Forest (HOG)
        p = os.path.join(self.models_dir, "best_rf.joblib")
        if os.path.exists(p):
            self.loaded_models["Random Forest (HOG)"] = {"model": joblib.load(p), "type": "sklearn", "feat_type": "hog"}

        # 5. Lightweight CNN
        p = os.path.join(self.models_dir, "cnn_model.pth")
        if os.path.exists(p):
            cnn = SymbolCNN(num_classes=self.num_classes)
            cnn.load_state_dict(torch.load(p, map_location="cpu"))
            cnn.eval()
            self.loaded_models["Lightweight 2D CNN"] = {"model": cnn, "type": "pytorch", "feat_type": "tensor"}

    def get_available_model_names(self):
        return list(self.loaded_models.keys())

    def predict(self, img_input, model_name="SVM (HOG)", top_k=3):
        """
        Run end-to-end inference on a single image.
        
        Parameters
        ----------
        img_input : PIL.Image or np.ndarray
        model_name : str, chosen model
        top_k : int, number of top predictions to return
        
        Returns
        -------
        result : dict containing:
            - top_predictions: list of {rank, class_name, latex, confidence, category}
            - best_prediction: top-1 dict
            - intermediates: dict of intermediate images
            - hog_image: HOG vector field visualization
        """
        if model_name not in self.loaded_models:
            # Fallback to first available model
            model_name = list(self.loaded_models.keys())[0]

        model_info = self.loaded_models[model_name]
        m = model_info["model"]
        m_type = model_info["type"]
        feat_type = model_info["feat_type"]

        # 1. Preprocess Image
        norm_img, intermediates = preprocess_single_image(img_input, target_size=(32, 32), return_intermediates=True)

        # 2. Generate HOG Visualization for explainability
        _, hog_vis = get_hog_visualization(norm_img)

        # 3. Extract Features
        if feat_type == "raw":
            feats = self.feat_pipe.transform(norm_img[np.newaxis, ...], feature_type="raw")
        elif feat_type == "hog":
            feats = self.feat_pipe.transform(norm_img[np.newaxis, ...], feature_type="hog")
        elif feat_type == "tensor":
            feats = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()

        # 4. Predict Probabilities
        if m_type == "sklearn":
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(feats)[0]
            else:
                # Decision function / softmax fallback
                df = m.decision_function(feats)[0]
                exp_df = np.exp(df - np.max(df))
                probs = exp_df / exp_df.sum()
        elif m_type == "pytorch":
            with torch.no_grad():
                logits = m(feats)
                probs = torch.softmax(logits, dim=1).numpy()[0]

        # 5. Extract Top-K Ranked Predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = []

        for rank, idx in enumerate(top_indices, 1):
            c_name = self.class_names[idx]
            info = self.class_stats.get(c_name, {})
            top_predictions.append({
                "rank": rank,
                "class_index": int(idx),
                "class_name": c_name,
                "latex": info.get("latex", c_name),
                "confidence": float(probs[idx]),
                "confidence_percent": float(probs[idx] * 100.0),
                "category": info.get("category", "symbol")
            })

        return {
            "model_used": model_name,
            "best_prediction": top_predictions[0],
            "top_predictions": top_predictions,
            "intermediates": intermediates,
            "hog_image": hog_vis
        }


if __name__ == "__main__":
    predictor = SymbolPredictor()
    print("Available models for inference:", predictor.get_available_model_names())
    # Test on random 32x32 image
    test_img = np.zeros((32, 32), dtype=np.uint8)
    test_img[10:22, 14:18] = 255  # vertical stroke
    test_img[14:18, 10:22] = 255  # horizontal stroke (like a plus)
    res = predictor.predict(test_img, model_name=predictor.get_available_model_names()[0])
    print("Test inference result:")
    print("Best prediction:", res["best_prediction"])
    print("Top 3:", res["top_predictions"])
