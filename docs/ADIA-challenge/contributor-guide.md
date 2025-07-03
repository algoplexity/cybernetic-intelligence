
---

# 🧭 Contributor Onboarding Guide

### ADIA Structural Break Challenge — Public Solution Architecture (CLv9-ECA Restricted Subset)

Welcome to the **Algoplexity contributor guide** for the ADIA Structural Break Detection challenge. This document introduces the architecture of our solution, describes each component's role, and provides clear entry points for you to contribute meaningfully — without requiring access to proprietary symbolic–latent research.

---

## 📦 System Overview

We model structural breaks in univariate time series using a **two-stage deep learning pipeline**:

1. **Pre-train** a dynamics-aware encoder on synthetic symbolic time series data derived from **Elementary Cellular Automata (ECA)**.
2. **Fine-tune** the encoder on labeled real-world time series data to detect breakpoints.

This results in a robust model that generalizes well to unseen structural regimes.

---

## 🧱 Architectural Layers

### 🟨 Notebook as a Self-Contained System

The entire solution runs as a single Python notebook executed by the ADIA Challenge Platform. It contains two callable entry points:

* `train(X_train, y_train, model_dir)`
* `infer(X_test, model_dir)`

The platform calls these functions and expects `infer()` to yield a sequence of scalar break predictions.

---

### 🧩 Major Components (C4 Level 2 Summary)

| Component              | Role                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| **Training Pipeline**  | Controls pre-training on ECA data and fine-tuning on real data.                  |
| **Inference Pipeline** | Loads the trained encoder and computes prediction scores from test data.         |
| **Core Library**       | Houses reusable modules: symbolic processors, model architectures, and encoders. |
| **Model Store**        | Filesystem layer used to save and load encoder weights and configuration.        |

---

## 🔄 Workflow Summary

### 📈 `train()` Function

1. **Pre-Training Stage**

   * Generate synthetic symbolic sequences using ECA rules.
   * Train a **DynamicalAutoencoder** with a reconstruction + classification loss.
   * Learn generalizable symbolic dynamics.

2. **Fine-Tuning Stage**

   * Process labeled real time series using `SeriesProcessor`.
   * Train a **StructuralBreakClassifier** to distinguish pre- and post-boundary regions using the encoder.
   * Save final model weights and config to `model_dir`.

---

### 📉 `infer()` Function

1. Load encoder weights and config via `EncoderLoader`.
2. For each test time series:

   * Transform pre- and post-boundary segments using `SeriesProcessor`.
   * Use `Fingerprinter` to generate high-level vector encodings.
   * Use `BreakScoreCalculator` to compute the **distance between pre- and post-fingerprints** (e.g. cosine distance).
3. Yield the score.

---

## 🛠️ Component-Level Roles (C4 Level 3 Highlights)

### 🔧 Core Services Library

| Class/Module                | Purpose                                                        |
| --------------------------- | -------------------------------------------------------------- |
| `PermutationSymbolizer`     | Converts numeric series into symbolic ordinal patterns         |
| `SeriesProcessor`           | Transforms full time series into windows of symbolic sequences |
| `TransformerEncoder`        | Learns vector representations (fingerprints) from sequences    |
| `DynamicalAutoencoder`      | Encodes + decodes sequences for unsupervised learning          |
| `StructuralBreakClassifier` | Binary classifier predicting breaks from two fingerprints      |

---

### ⚙️ Training Pipeline

| Class                      | Description                                                 |
| -------------------------- | ----------------------------------------------------------- |
| `ECADataGenerator`         | Produces labeled symbolic sequences using chaotic ECA rules |
| `MDLPreTrainer`            | Trains the DynamicalAutoencoder using a dual-loss signal    |
| `BreakClassifierFinetuner` | Fine-tunes the break classifier on real labeled data        |
| `EncoderSaver`             | Persists model weights and hyperparameters                  |

---

### ⚙️ Inference Pipeline

| Class                  | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `EncoderLoader`        | Loads model and config from Model Store                          |
| `Fingerprinter`        | Uses `SeriesProcessor` + encoder to create symbolic fingerprints |
| `BreakScoreCalculator` | Compares before/after fingerprints to produce final break score  |

---

## 🧪 Data Format (as provided by ADIA)

* `X_train`: `pd.DataFrame` with `MultiIndex [id, time]`, columns: `value`, `period`
* `y_train`: `pd.Series` with `id → bool` indicating break presence
* `X_test`: `List[pd.DataFrame]`, one per time series

---

## 🎯 Contribution Opportunities

You are welcome to contribute to the following areas without IP conflict:

1. **Improve ECA Sampling Logic**

   * Try alternate chaotic rules
   * Vary simulation width or depth

2. **Experiment with Alternative Fingerprinting Methods**

   * Swap `TransformerEncoder` with simpler architectures (e.g., GRU, CNN)

3. **Enhance Preprocessing**

   * Try Z-score normalization or smoothing in `SeriesProcessor`

4. **Tune Classifier Heads or Loss Functions**

   * Replace BCE with focal loss or hinge loss

5. **Modularization / Engineering Improvements**

   * Improve logging, error handling, or reproducibility features

6. **Add Test-Time Augmentations**

   * Add dropout, ensemble averaging, or segment perturbations

---

## 📎 Final Notes

* Please fork the project from the `algoplexity.github.io` notebook template or request edit access via the internal repo.
* Include comments on any experimental changes to allow reproducibility.
* Contributions will be reviewed based on clarity, performance gains, and maintainability.

