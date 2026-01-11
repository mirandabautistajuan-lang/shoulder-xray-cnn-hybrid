# calcifying-tendinopathy

**Artificial intelligence for detection of calcifying tendinopathy of the 
rotator cuff (CTRC) on conventional shoulder radiographs.**

This repository contains the experimental pipeline (Jupyter notebooks) 
used to develop, evaluate, and interpret AI models for **binary 
classification of shoulder X-rays** (*CTRC present vs absent*), including 
a systematic comparison between **end-to-end deep learning** and **hybrid 
CNN–machine learning** approaches.

---

## Project summary

Calcifying tendinopathy of the rotator cuff (CTRC) is a common cause of 
shoulder pain and functional limitation. **Conventional radiography** is 
the first-line imaging technique for identifying calcific deposits, but 
detection can be subtle and projection-dependent. This project develops an 
AI-based framework to detect CTRC on standard shoulder radiographs and 
compares two modeling strategies:

- **End-to-end CNN** (deep learning classifier).
- **Hybrid CNN–ML** (deep feature embeddings + classical ML classifier).

The repository is designed to support reproducible experimentation and 
transparent reporting for an academic manuscript.

---

## Main objective

Develop and evaluate an AI framework for classifying shoulder radiographs 
according to CTRC presence/absence, and compare **diagnostic 
performance**, **computational efficiency**, and **interpretability** 
between end-to-end CNN and hybrid CNN–ML models.

---

## Specific objectives

- Train and evaluate a fine-tuned CNN model for CTRC classification.
- Extract deep embeddings from the trained CNN and train classical ML 
models on these representations.
- Compare end-to-end CNN vs hybrid CNN–ML using ROC-based metrics and 
paired statistical testing.
- Provide model interpretability analyses using:
  - **Grad-CAM** for CNN predictions
  - **SHAP** for hybrid model feature contributions

---

## Methods (high level)

### End-to-end deep learning (classification)

- CNN backbone: **VGG19** pretrained on ImageNet (transfer learning / 
fine-tuning)
- Head: **Global Max Pooling** + **sigmoid** classifier
- Training: **stratified 5-fold cross-validation**
- Evaluation: independent balanced test set, ROC analysis, and DeLong test 
(paired comparison)

### Hybrid CNN–ML

- Feature extraction: activation maps from the CNN backbone (last 
convolutional block) aggregated into fixed-length embeddings
- Classical ML classifiers: logistic regression, SVM, KNN, decision tree, 
random forest
- Feature processing: correlation filtering, feature selection, 
scaling/encoding, cross-validated grid search

### Explainability

- **Grad-CAM**: qualitative localization of regions driving CNN 
predictions
- **SHAP**: feature-level attribution for the hybrid models

---

## Repository structure

### Notebooks (recommended order)

1. **00_environment_setup.ipynb**  
   Environment preparation, library checks, and global configuration.

2. **01_eda_dataset_statistics.ipynb**  
   Descriptive statistics and dataset characterization 
(demographics/technical variables).

3. **02_train_cnn_model.ipynb**  
   End-to-end CNN training (VGG19) with stratified cross-validation.

4. **03_extract_cnn_embeddings_for_ml.ipynb**  
   Deep feature extraction / embedding generation from the trained CNN for 
hybrid modeling.

5. **04_train_ml_on_embeddings.ipynb**  
   Training and optimization of classical ML classifiers on deep 
embeddings (+ optional metadata).

6. **05_eval_baseline_cnn_vs_ml.ipynb**  
   Comparative evaluation: end-to-end CNN vs best hybrid CNN–ML model, 
including ROC analysis and statistical testing.

---

## Ethics, privacy, and data governance

This repository provides **code and workflow only**. **No patient data is 
included**.

All medical imaging and associated metadata used in this project were 
handled under appropriate institutional approvals and governance 
procedures, in line with applicable ethical standards and data protection 
regulations. If you aim to reproduce these experiments, you must ensure 
compliance with your local institutional policies and legal requirements, 
and you should not upload any sensitive medical data to public 
repositories.

---

## Environment

Typical environment used in this project:

- Python 3.10 (Jupyter notebooks)

Key Python libraries commonly used in this workflow include:

- NumPy, Pandas
- TensorFlow / Keras
- scikit-learn
- OpenCV
- Matplotlib
- pydicom (if reproducing DICOM preprocessing outside this repository)
- tqdm

> Tip: keep notebook outputs cleared before pushing to GitHub to avoid 
rendering limits.

---

## Quick start (example)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

