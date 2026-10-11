"""
Feature Engineering Module
Handwritten Symbol Classifier

Extracts and compares two complementary representations:
1. Representation A (Raw Pixels): Flattened 32x32 = 1,024 continuous intensity values.
2. Representation B (HOG - Histogram of Oriented Gradients):
   - 9 orientation bins (classic Dalal-Triggs standard, 20 deg per bin)
   - 8x8 pixels per cell
   - 2x2 cells per block
   - L2-Hys block normalization
   - Yields 324 structured geometric edge features invariant to illumination and minor stroke thickness.

Includes:
- Batch HOG extraction with optional parallel execution
- Leakage-safe feature scaling using StandardScaler fitted exclusively on train data
- HOG feature visualizer for interpretability and Streamlit dashboard
"""

import os
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


class FeaturePipeline:
    """
    Feature pipeline providing Raw Pixel and HOG feature extraction
    with strict train-fitted scaling to prevent data leakage.
    """
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.scaler_raw = StandardScaler()
        self.scaler_hog = StandardScaler()
        self.is_fitted = False

    def extract_raw_features(self, X_images):
        """
        Flatten (N, 32, 32) images to (N, 1024) vectors.
        """
        N = X_images.shape[0]
        return X_images.reshape(N, -1).astype(np.float32)

    def extract_single_hog(self, img_2d, visualize=False):
        """
        Extract HOG feature for a single 32x32 2D image.
        """
        if visualize:
            feat, hog_img = hog(
                img_2d,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm="L2-Hys",
                visualize=True
            )
            return feat.astype(np.float32), hog_img
        else:
            feat = hog(
                img_2d,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm="L2-Hys",
                visualize=False
            )
            return feat.astype(np.float32)

    def extract_hog_features(self, X_images):
        """
        Extract HOG features for a batch of (N, 32, 32) images.
        """
        N = X_images.shape[0]
        hog_features = []
        for i in range(N):
            feat = self.extract_single_hog(X_images[i], visualize=False)
            hog_features.append(feat)
        return np.array(hog_features, dtype=np.float32)

    def fit(self, X_train_images):
        """
        Fit standard scalers strictly on training images.
        """
        raw_train = self.extract_raw_features(X_train_images)
        self.scaler_raw.fit(raw_train)

        hog_train = self.extract_hog_features(X_train_images)
        self.scaler_hog.fit(hog_train)

        self.is_fitted = True
        return self

    def transform(self, X_images, feature_type="hog", scale=True):
        """
        Transform images to specified feature representation.
        
        Parameters
        ----------
        X_images : np.ndarray, shape (N, 32, 32)
        feature_type : str, 'raw' or 'hog'
        scale : bool, whether to apply trained StandardScaler
        """
        if not self.is_fitted and scale:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform with scale=True!")

        if feature_type == "raw":
            feats = self.extract_raw_features(X_images)
            if scale:
                feats = self.scaler_raw.transform(feats)
        elif feature_type == "hog":
            feats = self.extract_hog_features(X_images)
            if scale:
                feats = self.scaler_hog.transform(feats)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}. Use 'raw' or 'hog'.")

        return feats

    def save(self, filepath="models/feature_pipeline.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Feature pipeline saved to {filepath}")

    @staticmethod
    def load(filepath="models/feature_pipeline.joblib"):
        return joblib.load(filepath)


def get_hog_visualization(img_2d):
    """
    Utility for Streamlit UI to visualize HOG orientation vectors.
    """
    feat, hog_img = hog(
        img_2d,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )
    return feat, hog_img


if __name__ == "__main__":
    from src.data_preprocessing import load_processed_dataset
    if os.path.exists("data/processed/math_symbols_dataset.npz"):
        X_train, y_train, _, _, _, _, _, _, _ = load_processed_dataset()
        print(f"Loaded train set: {X_train.shape}")
        pipeline = FeaturePipeline()
        pipeline.fit(X_train[:100])
        h_feats = pipeline.transform(X_train[:10], feature_type="hog")
        print(f"Sample HOG features shape: {h_feats.shape}")
