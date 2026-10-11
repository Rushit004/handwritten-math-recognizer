"""
Data Preprocessing Module
Handwritten Symbol Classifier

Handles:
- Tar archive inspection and selective subset extraction
- Image cleaning, polarity standardization (white stroke on dark background)
- Centering, bounding-box cropping, and aspect-ratio preserving resizing to 32x32
- Duplicate hash detection & corrupt image filtering
- Stratified Train / Validation / Test splitting (70 / 15 / 15) with fixed random seed
- Full vocabulary: 47 classes (30 Math Symbols + 10 Digits [0-9] + 6 Variables [x, y, z, a, b, n] + Equals [=])
"""

import os
import io
import json
import tarfile
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


# Comprehensive 47 Mathematical, Special, Numeric & Algebraic Classes
SELECTED_SYMBOLS = [
    # 1. Arithmetic Operators
    {"symbol_id": 196, "latex": "+", "name": "plus", "category": "arithmetic"},
    {"symbol_id": 195, "latex": "-", "name": "minus", "category": "arithmetic"},
    {"symbol_id": 513, "latex": "\\times", "name": "times", "category": "arithmetic"},
    {"symbol_id": 526, "latex": "\\div", "name": "div", "category": "arithmetic"},
    {"symbol_id": 524, "latex": "\\pm", "name": "plus_minus", "category": "arithmetic"},
    {"symbol_id": 9999, "latex": "=", "name": "equals", "category": "arithmetic"},

    # 2. Relational Operators
    {"symbol_id": 187, "latex": "<", "name": "less_than", "category": "relational"},
    {"symbol_id": 188, "latex": ">", "name": "greater_than", "category": "relational"},
    {"symbol_id": 185, "latex": "\\leq", "name": "less_equal", "category": "relational"},
    {"symbol_id": 186, "latex": "\\geq", "name": "greater_equal", "category": "relational"},
    {"symbol_id": 698, "latex": "\\neq", "name": "not_equal", "category": "relational"},

    # 3. Calculus & Analysis
    {"symbol_id": 960, "latex": "\\sqrt{}", "name": "sqrt", "category": "calculus"},
    {"symbol_id": 183, "latex": "\\int", "name": "integral", "category": "calculus"},
    {"symbol_id": 88,  "latex": "\\sum", "name": "sum", "category": "calculus"},
    {"symbol_id": 574, "latex": "\\prod", "name": "product", "category": "calculus"},
    {"symbol_id": 944, "latex": "\\infty", "name": "infinity", "category": "calculus"},
    {"symbol_id": 182, "latex": "\\partial", "name": "partial", "category": "calculus"},

    # 4. Greek Mathematical Letters
    {"symbol_id": 82,  "latex": "\\alpha", "name": "alpha", "category": "greek"},
    {"symbol_id": 87,  "latex": "\\beta", "name": "beta", "category": "greek"},
    {"symbol_id": 117, "latex": "\\gamma", "name": "gamma", "category": "greek"},
    {"symbol_id": 151, "latex": "\\delta", "name": "delta", "category": "greek"},
    {"symbol_id": 152, "latex": "\\Delta", "name": "Delta_cap", "category": "greek"},
    {"symbol_id": 81,  "latex": "\\pi", "name": "pi", "category": "greek"},
    {"symbol_id": 155, "latex": "\\theta", "name": "theta", "category": "greek"},
    {"symbol_id": 162, "latex": "\\lambda", "name": "lambda", "category": "greek"},
    {"symbol_id": 89,  "latex": "\\sigma", "name": "sigma", "category": "greek"},

    # 5. Set Theory & Logic
    {"symbol_id": 888, "latex": "\\in", "name": "in_element", "category": "set_logic"},
    {"symbol_id": 189, "latex": "\\subset", "name": "subset", "category": "set_logic"},
    {"symbol_id": 59,  "latex": "\\rightarrow", "name": "rightarrow", "category": "set_logic"},

    # 6. Delimiters
    {"symbol_id": 918, "latex": "[", "name": "left_bracket", "category": "delimiters"},
    {"symbol_id": 919, "latex": "]", "name": "right_bracket", "category": "delimiters"},

    # 7. Arabic Digits (0 - 9)
    {"symbol_id": 70, "latex": "0", "name": "digit_0", "category": "digits"},
    {"symbol_id": 71, "latex": "1", "name": "digit_1", "category": "digits"},
    {"symbol_id": 72, "latex": "2", "name": "digit_2", "category": "digits"},
    {"symbol_id": 73, "latex": "3", "name": "digit_3", "category": "digits"},
    {"symbol_id": 74, "latex": "4", "name": "digit_4", "category": "digits"},
    {"symbol_id": 75, "latex": "5", "name": "digit_5", "category": "digits"},
    {"symbol_id": 76, "latex": "6", "name": "digit_6", "category": "digits"},
    {"symbol_id": 77, "latex": "7", "name": "digit_7", "category": "digits"},
    {"symbol_id": 78, "latex": "8", "name": "digit_8", "category": "digits"},
    {"symbol_id": 79, "latex": "9", "name": "digit_9", "category": "digits"},

    # 8. Core Algebraic Variables
    {"symbol_id": 113, "latex": "x", "name": "var_x", "category": "variables"},
    {"symbol_id": 114, "latex": "y", "name": "var_y", "category": "variables"},
    {"symbol_id": 115, "latex": "z", "name": "var_z", "category": "variables"},
    {"symbol_id": 90,  "latex": "a", "name": "var_a", "category": "variables"},
    {"symbol_id": 91,  "latex": "b", "name": "var_b", "category": "variables"},
    {"symbol_id": 103, "latex": "n", "name": "var_n", "category": "variables"},
]


def preprocess_single_image(img_input, target_size=(32, 32), return_intermediates=False):
    """
    Standardize an input image to a centered 32x32 grayscale image with white strokes on black background.
    """
    if isinstance(img_input, Image.Image):
        if img_input.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img_input.size, (255, 255, 255))
            background.paste(img_input, mask=img_input.split()[-1])
            img_arr = np.array(background.convert("L"))
        else:
            img_arr = np.array(img_input.convert("L"))
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 3:
            if img_input.shape[2] == 4:
                alpha = img_input[:, :, 3] / 255.0
                rgb = cv2.cvtColor(img_input[:, :, :3], cv2.COLOR_RGB2GRAY)
                img_arr = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
            else:
                img_arr = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
        else:
            img_arr = img_input.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(img_input)}")

    raw_gray = img_arr.copy()

    # Check polarity
    h, w = img_arr.shape
    corners = [img_arr[0, 0], img_arr[0, -1], img_arr[-1, 0], img_arr[-1, -1]]
    if np.mean(corners) > 127:
        inverted = 255 - img_arr
    else:
        inverted = img_arr.copy()

    smoothed = cv2.GaussianBlur(inverted, (3, 3), 0)
    _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        cropped = thresh[y:y+bh, x:x+bw]
        
        inner_max = int(min(target_size) * 0.75)  # 24 px
        scale = inner_max / max(bw, bh)
        new_w = max(1, int(bw * scale))
        new_h = max(1, int(bh * scale))
        
        resized_crop = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros(target_size, dtype=np.uint8)
        start_x = (target_size[1] - new_w) // 2
        start_y = (target_size[0] - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_crop
    else:
        canvas = cv2.resize(smoothed, target_size, interpolation=cv2.INTER_AREA)

    normalized = canvas.astype(np.float32) / 255.0

    if return_intermediates:
        intermediates = {
            "raw_gray": raw_gray,
            "inverted": inverted,
            "thresholded": thresh,
            "centered": canvas,
            "normalized": normalized
        }
        return normalized, intermediates

    return normalized


def extract_and_prepare_dataset(
    tar_path="data/raw/HASYv2.tar.bz2",
    labels_csv_path="data/raw/hasy-data-labels.csv",
    output_npz_path="data/processed/math_symbols_dataset.npz",
    output_meta_path="data/processed/metadata.json",
    sample_images_dir="data/processed/sample_images",
    random_state=42
):
    print(f"Loading labels from {labels_csv_path}...")
    labels_df = pd.read_csv(labels_csv_path)
    
    symbol_id_to_meta = {s["symbol_id"]: s for s in SELECTED_SYMBOLS}
    hasy_ids = [s["symbol_id"] for s in SELECTED_SYMBOLS if s["symbol_id"] != 9999]
    filtered_df = labels_df[labels_df["symbol_id"].isin(hasy_ids)].copy()
    
    print(f"Total matching records across {len(hasy_ids)} HASYv2 symbols: {len(filtered_df)}")

    filtered_df["class_name"] = filtered_df["symbol_id"].map(lambda sid: symbol_id_to_meta[sid]["name"])
    filtered_df["latex_cmd"] = filtered_df["symbol_id"].map(lambda sid: symbol_id_to_meta[sid]["latex"])
    filtered_df["category"] = filtered_df["symbol_id"].map(lambda sid: symbol_id_to_meta[sid]["category"])

    os.makedirs(sample_images_dir, exist_ok=True)
    all_class_names = sorted([s["name"] for s in SELECTED_SYMBOLS])
    class_to_idx = {c: i for i, c in enumerate(all_class_names)}
    idx_to_class = {i: c for i, c in enumerate(all_class_names)}
    filtered_df["label_idx"] = filtered_df["class_name"].map(class_to_idx)

    sample_saved = {c: 0 for c in all_class_names}

    # Extract images from tar archive
    print(f"Extracting images from archive {tar_path}...")
    image_paths_set = set(filtered_df["path"].tolist())
    path_to_data = {}
    
    md5_hashes = set()
    duplicate_count = 0
    corrupt_count = 0

    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if member.name in image_paths_set:
                try:
                    f = tar.extractfile(member)
                    raw_bytes = f.read()
                    
                    file_hash = hashlib.md5(raw_bytes).hexdigest()
                    if file_hash in md5_hashes:
                        duplicate_count += 1
                    else:
                        md5_hashes.add(file_hash)

                    img = Image.open(io.BytesIO(raw_bytes))
                    path_to_data[member.name] = img
                except Exception:
                    corrupt_count += 1

    print(f"Extracted {len(path_to_data)} images. Found {duplicate_count} exact duplicates, {corrupt_count} corrupt.")

    processed_images = []
    valid_rows = []
    minus_images = []

    for _, row in filtered_df.iterrows():
        p = row["path"]
        if p in path_to_data:
            img = path_to_data[p]
            c_name = row["class_name"]
            
            if sample_saved[c_name] < 3:
                sample_folder = os.path.join(sample_images_dir, c_name)
                os.makedirs(sample_folder, exist_ok=True)
                img.save(os.path.join(sample_folder, f"sample_{sample_saved[c_name]}.png"))
                sample_saved[c_name] += 1
                
            norm_img = preprocess_single_image(img, target_size=(32, 32))
            processed_images.append(norm_img)
            valid_rows.append(row.to_dict())

            if c_name == "minus":
                minus_images.append(norm_img)

    # Synthesize authentic handwritten equals signs (=) from pairs of minus strokes
    print("Synthesizing handwritten equals signs (=)...")
    np.random.seed(random_state)
    equals_rows = []
    equals_images = []
    num_equals = 220

    equals_folder = os.path.join(sample_images_dir, "equals")
    os.makedirs(equals_folder, exist_ok=True)

    for i in range(num_equals):
        # Pick two random minus strokes
        idx1 = np.random.randint(0, len(minus_images))
        idx2 = np.random.randint(0, len(minus_images))
        m1 = (minus_images[idx1] * 255).astype(np.uint8)
        m2 = (minus_images[idx2] * 255).astype(np.uint8)

        # Scale strokes slightly vertically
        canvas_eq = np.zeros((32, 32), dtype=np.uint8)
        gap = np.random.randint(7, 11)
        offset_x = np.random.randint(-2, 3)
        
        # Place top bar around y=11, bottom bar around y=11+gap
        y_top = 16 - (gap // 2) - 2
        y_bot = 16 + (gap // 2)

        # Extract active strokes
        c1 = cv2.findNonZero(m1)
        c2 = cv2.findNonZero(m2)
        if c1 is not None and c2 is not None:
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            crop1 = m1[y1:y1+h1, x1:x1+w1]
            crop2 = m2[y2:y2+w2, x2:x2+w2]
            
            # Place onto canvas
            w_target = np.random.randint(14, 22)
            c1_res = cv2.resize(crop1, (w_target, max(2, h1)), interpolation=cv2.INTER_AREA)
            c2_res = cv2.resize(crop2, (w_target, max(2, h2)), interpolation=cv2.INTER_AREA)

            start_x = (32 - w_target) // 2 + offset_x
            start_x = max(1, min(31 - w_target, start_x))

            canvas_eq[max(0, y_top):min(32, y_top+c1_res.shape[0]), start_x:start_x+w_target] = c1_res[:min(32-y_top, c1_res.shape[0]), :]
            canvas_eq[max(0, y_bot):min(32, y_bot+c2_res.shape[0]), start_x:start_x+w_target] = c2_res[:min(32-y_bot, c2_res.shape[0]), :]
        else:
            # Fallback direct parallel bars
            canvas_eq[11:13, 6:26] = 255
            canvas_eq[19:21, 6:26] = 255

        norm_eq = canvas_eq.astype(np.float32) / 255.0
        equals_images.append(norm_eq)
        equals_rows.append({
            "path": f"synthetic/equals_{i}.png",
            "symbol_id": 9999,
            "latex": "=",
            "class_name": "equals",
            "latex_cmd": "=",
            "category": "arithmetic",
            "label_idx": class_to_idx["equals"]
        })

        if i < 3:
            # Save sample PNG
            sample_png = Image.fromarray(255 - canvas_eq)
            sample_png.save(os.path.join(equals_folder, f"sample_{i}.png"))

    all_processed = processed_images + equals_images
    all_rows = valid_rows + equals_rows

    X_all = np.array(all_processed, dtype=np.float32)
    valid_df = pd.DataFrame(all_rows)
    y_all = valid_df["label_idx"].values
    latex_all = valid_df["latex_cmd"].values
    class_all = valid_df["class_name"].values

    print(f"Total dataset records: {len(X_all)}, Total classes: {len(all_class_names)}")

    # Stratified Train/Val/Test Split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp, latex_train, latex_temp, class_train, class_temp = train_test_split(
        X_all, y_all, latex_all, class_all,
        test_size=0.30,
        random_state=random_state,
        stratify=y_all
    )

    X_val, X_test, y_val, y_test, latex_val, latex_test, class_val, class_test = train_test_split(
        X_temp, y_temp, latex_temp, class_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Splits created: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # Save to NPZ
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        X_train=X_train, y_train=y_train, latex_train=latex_train, class_train=class_train,
        X_val=X_val, y_val=y_val, latex_val=latex_val, class_val=class_val,
        X_test=X_test, y_test=y_test, latex_test=latex_test, class_test=class_test
    )
    print(f"Saved dataset archive to {output_npz_path}")

    # Save metadata dictionary
    class_stats = {}
    for c_idx, c_name in idx_to_class.items():
        sym_info = next(s for s in SELECTED_SYMBOLS if s["name"] == c_name)
        n_train = int(np.sum(y_train == c_idx))
        n_val = int(np.sum(y_val == c_idx))
        n_test = int(np.sum(y_test == c_idx))
        class_stats[c_name] = {
            "index": c_idx,
            "latex": sym_info["latex"],
            "symbol_id": sym_info["symbol_id"],
            "category": sym_info["category"],
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "total_samples": n_train + n_val + n_test
        }

    metadata = {
        "dataset_name": "HASYv2 Mathematical, Alphanumeric & Special Symbols",
        "author": "Martin Thoma (KIT, 2017)",
        "license": "CC0 1.0 Universal",
        "num_classes": len(all_class_names),
        "total_samples": len(y_all),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "image_shape": [32, 32],
        "duplicate_count": duplicate_count,
        "corrupt_count": corrupt_count,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_statistics": class_stats
    }

    with open(output_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_meta_path}")
    return metadata


def load_processed_dataset(npz_path="data/processed/math_symbols_dataset.npz"):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["X_train"], data["y_train"], data["class_train"],
        data["X_val"], data["y_val"], data["class_val"],
        data["X_test"], data["y_test"], data["class_test"]
    )


if __name__ == "__main__":
    extract_and_prepare_dataset()
