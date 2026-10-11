"""
Evaluation & Error Analysis Module
Handwritten Symbol Classifier

Performs:
- Full evaluation of all models on the UNTOUCHED test set (5,745 samples)
- Multi-metric reporting: Accuracy, Balanced Accuracy, Precision, Recall, Macro F1, Weighted F1
- Confusion Matrix generation & visualization
- Comprehensive Error Analysis:
  - Top confused symbol pairs identification
  - Visual gallery of actual misclassified images with True vs Predicted labels and confidence
  - Structural/morphological explanations for errors
- Inference latency & model artifact size measurement
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

import torch
import torch.nn as nn

from src.data_preprocessing import load_processed_dataset
from src.feature_engineering import FeaturePipeline
from src.train import SymbolCNN


def run_evaluation_pipeline(
    dataset_path="data/processed/math_symbols_dataset.npz",
    metadata_path="data/processed/metadata.json"
):
    print("=" * 60)
    print("STARTING TEST EVALUATION & ERROR ANALYSIS")
    print("=" * 60)

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/confusion_matrix", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load Data & Metadata
    X_train, y_train, class_train, X_val, y_val, class_val, X_test, y_test, class_test = load_processed_dataset(dataset_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_names = [metadata["idx_to_class"][str(i)] for i in range(metadata["num_classes"])]
    latex_map = {metadata["idx_to_class"][str(i)]: metadata["class_statistics"][metadata["idx_to_class"][str(i)]]["latex"] for i in range(metadata["num_classes"])}
    num_classes = len(class_names)
    print(f"Loaded Test Set: {len(y_test)} samples across {num_classes} classes.")

    # 2. Load Feature Pipeline & Extract Test Features
    feat_pipe = FeaturePipeline.load("models/feature_pipeline.joblib")
    raw_test = feat_pipe.transform(X_test, feature_type="raw")
    hog_test = feat_pipe.transform(X_test, feature_type="hog")

    # 3. Load Trained Models
    models = {}
    if os.path.exists("models/baseline_logreg_raw.joblib"):
        models["Baseline (LogReg Raw)"] = {"model": joblib.load("models/baseline_logreg_raw.joblib"), "type": "sklearn", "features": raw_test}
    if os.path.exists("models/baseline_logreg_hog.joblib"):
        models["Baseline (LogReg HOG)"] = {"model": joblib.load("models/baseline_logreg_hog.joblib"), "type": "sklearn", "features": hog_test}
    if os.path.exists("models/best_svm.joblib"):
        models["SVM (HOG)"] = {"model": joblib.load("models/best_svm.joblib"), "type": "sklearn", "features": hog_test}
    if os.path.exists("models/best_rf.joblib"):
        models["Random Forest (HOG)"] = {"model": joblib.load("models/best_rf.joblib"), "type": "sklearn", "features": hog_test}

    # Load PyTorch CNN
    if os.path.exists("models/cnn_model.pth"):
        cnn = SymbolCNN(num_classes=num_classes)
        cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
        cnn.eval()
        models["Lightweight 2D CNN"] = {"model": cnn, "type": "pytorch", "features": X_test}

    # 4. Evaluate Each Model
    metrics_summary = {}
    predictions_dict = {}
    probabilities_dict = {}

    for name, item in models.items():
        print(f"\nEvaluating {name} on Test Set...")
        m = item["model"]
        m_type = item["type"]
        feats = item["features"]

        t0 = time.time()
        if m_type == "sklearn":
            preds = m.predict(feats)
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(feats)
            else:
                probs = None
        elif m_type == "pytorch":
            test_tensor = torch.from_numpy(feats).unsqueeze(1)
            with torch.no_grad():
                logits = m(test_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
                preds = logits.argmax(dim=1).numpy()
        
        latency_ms = ((time.time() - t0) / len(y_test)) * 1000.0

        # Compute Metrics
        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

        predictions_dict[name] = preds
        probabilities_dict[name] = probs

        # Compute Model Size on Disk
        file_map = {
            "Baseline (LogReg Raw)": "models/baseline_logreg_raw.joblib",
            "Baseline (LogReg HOG)": "models/baseline_logreg_hog.joblib",
            "SVM (HOG)": "models/best_svm.joblib",
            "Random Forest (HOG)": "models/best_rf.joblib",
            "Lightweight 2D CNN": "models/cnn_model.pth"
        }
        fpath = file_map.get(name, "")
        size_mb = os.path.getsize(fpath) / (1024 * 1024) if os.path.exists(fpath) else 0.0

        metrics_summary[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_precision": float(prec_m),
            "macro_recall": float(rec_m),
            "macro_f1": float(f1_m),
            "weighted_precision": float(prec_w),
            "weighted_recall": float(rec_w),
            "weighted_f1": float(f1_w),
            "inference_latency_ms": float(latency_ms),
            "model_size_mb": float(size_mb)
        }

        print(f"  Accuracy: {acc:.4f} | Bal Acc: {bal_acc:.4f} | Macro F1: {f1_m:.4f} | Latency: {latency_ms:.2f}ms")

    # Save metrics JSON
    with open("results/metrics/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Convert to DataFrame for display & save
    summary_df = pd.DataFrame(metrics_summary).T
    summary_df.to_csv("results/metrics/metrics_summary.csv")
    print("\nTest Evaluation Summary:")
    print(summary_df[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "inference_latency_ms", "model_size_mb"]].to_string())

    # 5. Class-wise Performance for Best Model
    best_model_name = max(metrics_summary.keys(), key=lambda k: metrics_summary[k]["macro_f1"])
    best_preds = predictions_dict[best_model_name]
    print(f"\nBest Model by Macro F1: {best_model_name}")

    cls_report = classification_report(y_test, best_preds, target_names=class_names, output_dict=True, zero_division=0)
    cls_df = pd.DataFrame(cls_report).T
    cls_df.to_csv("results/metrics/class_wise_metrics.csv")

    # 6. Confusion Matrix Generation
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, cmap="Blues", fmt="d",
        xticklabels=[f"{c} ({latex_map[c]})" for c in class_names],
        yticklabels=[f"{c} ({latex_map[c]})" for c in class_names]
    )
    plt.title(f"Confusion Matrix: {best_model_name} on Test Set ({acc:.2%} Acc)", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix/confusion_matrix_best.png", dpi=200)
    plt.close()
    print("Saved Confusion Matrix to results/confusion_matrix/confusion_matrix_best.png")

    # 7. Error Analysis: Top Confused Pairs & Actual Misclassified Images
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "true_latex": latex_map[class_names[i]],
                    "pred_class": class_names[j],
                    "pred_latex": latex_map[class_names[j]],
                    "error_count": int(cm[i, j]),
                    "true_class_support": int(np.sum(y_test == i))
                })

    confused_pairs = sorted(confused_pairs, key=lambda x: x["error_count"], reverse=True)
    with open("results/confusion_matrix/top_confused_pairs.json", "w") as f:
        json.dump(confused_pairs[:20], f, indent=2)

    print("\nTop 5 Confused Symbol Pairs:")
    for p in confused_pairs[:5]:
        print(f"  True: {p['true_class']} ({p['true_latex']}) -> Pred: {p['pred_class']} ({p['pred_latex']}): {p['error_count']} errors")

    # 8. Visual Gallery of Actual Misclassifications
    incorrect_indices = np.where(y_test != best_preds)[0]
    best_probs = probabilities_dict.get(best_model_name, None)

    if len(incorrect_indices) > 0:
        sample_errors = incorrect_indices[:min(12, len(incorrect_indices))]
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        axes = axes.flatten()

        for idx, err_i in enumerate(sample_errors):
            true_lbl = y_test[err_i]
            pred_lbl = best_preds[err_i]
            img = X_test[err_i]
            conf = best_probs[err_i, pred_lbl] * 100.0 if best_probs is not None else 0.0

            t_ltx = latex_map[class_names[true_lbl]].replace(r"\sqrt{}", r"\sqrt{\ }")
            p_ltx = latex_map[class_names[pred_lbl]].replace(r"\sqrt{}", r"\sqrt{\ }")
            axes[idx].imshow(img, cmap="gray")
            axes[idx].set_title(
                f"True: {class_names[true_lbl]} (${t_ltx}$)\nPred: {class_names[pred_lbl]} (${p_ltx}$)\nConf: {conf:.1f}%",
                fontsize=10, fontweight="bold", color="red"
            )
            axes[idx].axis("off")

        for idx in range(len(sample_errors), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(f"Error Analysis: Sample Misclassified Symbols ({best_model_name})", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig("results/figures/misclassified_examples.png", dpi=200)
        plt.close()
        print("Saved Misclassified Examples to results/figures/misclassified_examples.png")

    print("\n" + "=" * 60)
    print("EVALUATION & ERROR ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation_pipeline()
