"""
Training Pipeline Module
Handwritten Symbol Classifier

Trains and validates multiple machine learning paradigms:
1. Baseline: Logistic Regression (L2) - Raw Pixels vs HOG
2. Model 1: Support Vector Machine (Linear & RBF kernel) on HOG with Grid Search
3. Model 2: Random Forest Classifier on HOG with hyperparameter tuning
4. Model 3: Lightweight 2D CNN (PyTorch) on 2D images with class weights & data augmentation

Persists models to models/ and training logs to results/metrics/
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.data_preprocessing import load_processed_dataset
from src.feature_engineering import FeaturePipeline


# ==========================================
# 1. PyTorch CNN Architecture & Augmentation
# ==========================================

class SymbolCNN(nn.Module):
    """
    Lightweight 2D Convolutional Neural Network for 32x32 handwritten symbol images.
    
    Architecture:
    - Conv Block 1: Conv2d(1->32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) [16x16]
    - Conv Block 2: Conv2d(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) [8x8]
    - Dropout(0.25)
    - FC 1: Linear(64*8*8 -> 128) -> ReLU -> Dropout(0.5)
    - FC 2: Linear(128 -> num_classes)
    """
    def __init__(self, num_classes=30):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
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


def augment_tensor_batch(images):
    """
    On-the-fly data augmentation:
    - Random small rotation (-10 to +10 degrees)
    - Random small translation (-2 to +2 pixels)
    - Applied during CNN training only
    """
    # Simple PyTorch affine translation/rotation
    device = images.device
    B = images.size(0)
    
    angles = (torch.rand(B, device=device) * 20.0 - 10.0) * (np.pi / 180.0)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    tx = (torch.rand(B, device=device) * 4.0 - 2.0) / 16.0
    ty = (torch.rand(B, device=device) * 4.0 - 2.0) / 16.0
    
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty
    
    grid = nn.functional.affine_grid(theta, images.size(), align_corners=False)
    augmented = nn.functional.grid_sample(images, grid, align_corners=False, mode="bilinear", padding_mode="zeros")
    return augmented


# ==========================================
# 2. Main Training Workflow
# ==========================================

def run_training_pipeline(dataset_path="data/processed/math_symbols_dataset.npz"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    
    print("=" * 60)
    print("STARTING HANDWRITTEN SYMBOL CLASSIFIER TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    X_train, y_train, class_train, X_val, y_val, class_val, X_test, y_test, class_test = load_processed_dataset(dataset_path)
    num_classes = len(np.unique(y_train))
    print(f"Data Loaded: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}, Classes={num_classes}")

    # Compute Analytical Class Weights (Stage 5: Class Imbalance)
    class_weights_np = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights_np)}
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    # 2. Feature Engineering: Fit Pipeline & Transform
    print("\n--- STAGE 6: Feature Engineering (Raw Pixels vs HOG) ---")
    feat_pipe = FeaturePipeline()
    feat_pipe.fit(X_train)
    feat_pipe.save("models/feature_pipeline.joblib")

    print("Extracting features for training set...")
    raw_train = feat_pipe.transform(X_train, feature_type="raw")
    raw_val = feat_pipe.transform(X_val, feature_type="raw")

    hog_train = feat_pipe.transform(X_train, feature_type="hog")
    hog_val = feat_pipe.transform(X_val, feature_type="hog")
    print(f"Feature Dimensions: Raw Pixels = {raw_train.shape[1]}, HOG Features = {hog_train.shape[1]}")

    results_summary = {}

    # ----------------------------------------------------
    # Baseline Model: Logistic Regression (Raw vs HOG)
    # ----------------------------------------------------
    print("\n--- STAGE 8: Baseline Models (Logistic Regression) ---")
    
    # Baseline A: Logistic Regression on Raw Pixels
    t0 = time.time()
    lr_raw = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    lr_raw.fit(raw_train, y_train)
    time_lr_raw = time.time() - t0
    val_pred_lr_raw = lr_raw.predict(raw_val)
    acc_lr_raw = accuracy_score(y_val, val_pred_lr_raw)
    f1_lr_raw = f1_score(y_val, val_pred_lr_raw, average="macro")
    print(f"[Baseline A - Raw Pixels] Val Acc: {acc_lr_raw:.4f}, Macro F1: {f1_lr_raw:.4f}, Train Time: {time_lr_raw:.2f}s")
    joblib.dump(lr_raw, "models/baseline_logreg_raw.joblib")

    # Baseline B: Logistic Regression on HOG Features
    t0 = time.time()
    lr_hog = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    lr_hog.fit(hog_train, y_train)
    time_lr_hog = time.time() - t0
    val_pred_lr_hog = lr_hog.predict(hog_val)
    acc_lr_hog = accuracy_score(y_val, val_pred_lr_hog)
    f1_lr_hog = f1_score(y_val, val_pred_lr_hog, average="macro")
    print(f"[Baseline B - HOG Features] Val Acc: {acc_lr_hog:.4f}, Macro F1: {f1_lr_hog:.4f}, Train Time: {time_lr_hog:.2f}s")
    joblib.dump(lr_hog, "models/baseline_logreg_hog.joblib")

    results_summary["Baseline_LogReg_Raw"] = {
        "val_accuracy": float(acc_lr_raw), "val_macro_f1": float(f1_lr_raw),
        "train_time_sec": float(time_lr_raw), "features": "Raw Pixels (1024-D)"
    }
    results_summary["Baseline_LogReg_HOG"] = {
        "val_accuracy": float(acc_lr_hog), "val_macro_f1": float(f1_lr_hog),
        "train_time_sec": float(time_lr_hog), "features": "HOG (324-D)"
    }

    # ----------------------------------------------------
    # Model 1: Support Vector Machine (SVM) + Tuning
    # ----------------------------------------------------
    print("\n--- STAGE 9 & 10: Model 1 - Support Vector Machine (SVM) ---")
    print("Performing Hyperparameter Tuning with 3-Fold Stratified CV on HOG features...")
    # Tune on representative stratified subset (6,000 samples) to ensure rapid CV convergence
    cv_idx = np.random.RandomState(42).choice(len(y_train), size=min(6000, len(y_train)), replace=False)
    svm_param_grid = {
        "C": [0.5, 2.0, 10.0],
        "kernel": ["linear", "rbf"]
    }
    svm_grid = GridSearchCV(
        SVC(class_weight="balanced", probability=True, random_state=42),
        svm_param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    svm_grid.fit(hog_train[cv_idx], y_train[cv_idx])
    tune_time_svm = time.time() - t0
    print(f"Best SVM Parameters found: {svm_grid.best_params_} (CV Macro F1: {svm_grid.best_score_:.4f})")

    # Fit final best SVM model on full training set
    print("Fitting best SVM model on full training set...")
    t0 = time.time()
    best_svm = SVC(**svm_grid.best_params_, class_weight="balanced", probability=True, random_state=42)
    best_svm.fit(hog_train, y_train)
    fit_time_svm = time.time() - t0
    val_pred_svm = best_svm.predict(hog_val)
    acc_svm = accuracy_score(y_val, val_pred_svm)
    f1_svm = f1_score(y_val, val_pred_svm, average="macro")
    print(f"[SVM Best] Val Acc: {acc_svm:.4f}, Macro F1: {f1_svm:.4f}, Total Time: {fit_time_svm:.2f}s")
    joblib.dump(best_svm, "models/best_svm.joblib")

    results_summary["SVM_HOG"] = {
        "val_accuracy": float(acc_svm), "val_macro_f1": float(f1_svm),
        "train_time_sec": float(fit_time_svm), "tuning_time_sec": float(tune_time_svm),
        "best_params": svm_grid.best_params_, "features": "HOG (324-D)"
    }

    # ----------------------------------------------------
    # Model 2: Random Forest Classifier + Tuning
    # ----------------------------------------------------
    print("\n--- STAGE 9 & 10: Model 2 - Random Forest Classifier ---")
    rf_param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [20, None],
        "min_samples_split": [2, 5]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    rf_grid.fit(hog_train[cv_idx], y_train[cv_idx])
    tune_time_rf = time.time() - t0
    print(f"Best Random Forest Parameters: {rf_grid.best_params_} (CV Macro F1: {rf_grid.best_score_:.4f})")

    print("Fitting best Random Forest model on full training set...")
    t0 = time.time()
    best_rf = RandomForestClassifier(**rf_grid.best_params_, class_weight="balanced", n_jobs=-1, random_state=42)
    best_rf.fit(hog_train, y_train)
    fit_time_rf = time.time() - t0
    val_pred_rf = best_rf.predict(hog_val)
    acc_rf = accuracy_score(y_val, val_pred_rf)
    f1_rf = f1_score(y_val, val_pred_rf, average="macro")
    print(f"[Random Forest Best] Val Acc: {acc_rf:.4f}, Macro F1: {f1_rf:.4f}, Total Time: {fit_time_rf:.2f}s")
    joblib.dump(best_rf, "models/best_rf.joblib")

    results_summary["Random_Forest_HOG"] = {
        "val_accuracy": float(acc_rf), "val_macro_f1": float(f1_rf),
        "train_time_sec": float(fit_time_rf), "tuning_time_sec": float(tune_time_rf),
        "best_params": rf_grid.best_params_, "features": "HOG (324-D)"
    }

    # ----------------------------------------------------
    # Model 3: Lightweight 2D CNN (PyTorch)
    # ----------------------------------------------------
    print("\n--- STAGE 9 & 10: Model 3 - Lightweight 2D CNN (PyTorch) ---")
    batch_size = 64
    epochs = 8
    
    # Prepare PyTorch Tensors (N, 1, 32, 32)
    train_tensor_x = torch.from_numpy(X_train).unsqueeze(1)
    train_tensor_y = torch.from_numpy(y_train).long()
    val_tensor_x = torch.from_numpy(X_val).unsqueeze(1)
    val_tensor_y = torch.from_numpy(y_val).long()

    train_ds = TensorDataset(train_tensor_x, train_tensor_y)
    val_ds = TensorDataset(val_tensor_x, val_tensor_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    cnn_model = SymbolCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        cnn_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Apply on-the-fly augmentation
            aug_x = augment_tensor_batch(batch_x)
            optimizer.zero_grad()
            outputs = cnn_model(aug_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        cnn_model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = cnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(cnn_model.state_dict(), "models/cnn_model.pth")

    time_cnn = time.time() - t0
    final_cnn_acc = history["val_acc"][-1]
    final_cnn_f1 = history["val_f1"][-1]
    print(f"[CNN Model] Best Val Loss: {best_val_loss:.4f}, Final Acc: {final_cnn_acc:.4f}, Train Time: {time_cnn:.2f}s")

    results_summary["CNN_2D"] = {
        "val_accuracy": float(final_cnn_acc), "val_macro_f1": float(final_cnn_f1),
        "best_val_loss": float(best_val_loss), "train_time_sec": float(time_cnn),
        "epochs": epochs, "features": "2D Image Tensors (32x32) with Data Augmentation"
    }

    # Save summary
    with open("results/metrics/training_validation_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    with open("results/metrics/cnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE! Artifacts successfully generated.")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
