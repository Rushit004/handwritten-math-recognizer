# Dataset Documentation: Handwritten Mathematical Symbols (HASYv2 Subset)

## 1. Dataset Overview
- **Dataset Name:** HASYv2 (HAtala SYmbols version 2)
- **Primary Author:** Martin Thoma, Karlsruhe Institute of Technology (KIT)
- **Source Repository:** [Zenodo Record 259444](https://zenodo.org/records/259444)
- **Publication Reference:** Thoma, M. (2017). *The HASYv2 dataset*. arXiv preprint [arXiv:1701.08380](https://arxiv.org/abs/1701.08380).
- **License:** **Creative Commons CC0 1.0 Universal (Public Domain Dedication)**
  - Fully permissible for open educational, university, and commercial applications without copyright restrictions.

## 2. Why HASYv2 Was Selected Over MNIST
1. **Curriculum Alignment:** MNIST contains only 10 Arabic digits (0–9), which is trivial and heavily saturated. HASYv2 was designed explicitly as the open mathematical counterpart to MNIST.
2. **Domain Richness:** Covers mathematical operators, relations, calculus notations, Greek alphabet glyphs, set operations, and delimiters.
3. **Genuine Real-World Properties:** Contains variations in stroke thickness, handwriting angle, imperfect centering, and natural class imbalance reflected in user writing frequencies.

## 3. Curated 30 Mathematical Symbol Classes
For this project, we curated a representative subset of **30 mathematical and special symbols**:

| ID | LaTeX Command | Rendered Symbol | Mathematical Category |
| :---: | :---: | :---: | :--- |
| 196 | `+` | $+$ | Arithmetic Addition |
| 195 | `-` | $-$ | Arithmetic Subtraction |
| 513 | `\times` | $\times$ | Multiplication |
| 526 | `\div` | $\div$ | Division |
| 524 | `\pm` | $\pm$ | Plus-Minus Operator |
| 187 | `<` | $<$ | Strictly Less Than |
| 188 | `>` | $>$ | Strictly Greater Than |
| 185 | `\leq` | $\le$ | Less Than or Equal |
| 186 | `\geq` | $\ge$ | Greater Than or Equal |
| 698 | `\neq` | $\ne$ | Inequality / Not Equal |
| 960 | `\sqrt{}` | $\sqrt{}$ | Square Root Radical |
| 183 | `\int` | $\int$ | Integral Operator |
| 88 | `\sum` | $\sum$ | Summation Operator |
| 574 | `\prod` | $\prod$ | Product Operator |
| 944 | `\infty` | $\infty$ | Infinity Symbol |
| 182 | `\partial` | $\partial$ | Partial Derivative |
| 82 | `\alpha` | $\alpha$ | Greek Alpha |
| 87 | `\beta` | $\beta$ | Greek Beta |
| 117 | `\gamma` | $\gamma$ | Greek Gamma |
| 151 | `\delta` | $\delta$ | Greek Delta (lowercase) |
| 152 | `\Delta` | $\Delta$ | Greek Delta (uppercase) |
| 81 | `\pi` | $\pi$ | Mathematical Constant Pi |
| 155 | `\theta` | $\theta$ | Greek Theta |
| 162 | `\lambda` | $\lambda$ | Greek Lambda |
| 89 | `\sigma` | $\sigma$ | Greek Sigma (lowercase) |
| 888 | `\in` | $\in$ | Set Membership Element |
| 189 | `\subset` | $\subset$ | Proper Subset |
| 59 | `\rightarrow` | $\rightarrow$ | Implication / Vector Arrow |
| 918 | `[` | $[$ | Left Square Bracket |
| 919 | `]` | $]$ | Right Square Bracket |

## 4. File Structure & Processed Storage
- `data/raw/HASYv2.tar.bz2`: Full original compressed archive from Zenodo.
- `data/raw/symbols.csv`: Metadata table mapping `symbol_id`, `latex`, and native counts.
- `data/raw/hasy-data-labels.csv`: Manifest of 168,233 image filepaths and their ground truth labels.
- `data/processed/math_symbols_dataset.npz`: Compressed NumPy archive containing:
  - `X_train`, `y_train`, `train_latex`
  - `X_val`, `y_val`, `val_latex`
  - `X_test`, `y_test`, `test_latex`
- `data/processed/metadata.json`: Class label index mapping, latex strings, and dataset statistics.
- `data/processed/sample_images/`: High-resolution representative PNGs of each class for quick preview and interactive testing.
