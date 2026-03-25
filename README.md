# cell-segmentation-watershed
Automatic cell segmentation using Watershed
# 🧬 Automatic Cell Segmentation using Watershed

## 📌 Project Overview
This project focuses on **automatic cell segmentation** from microscopy images using Python and image processing techniques.

The goal is to:
- Detect individual cells
- Separate overlapping cells
- Extract quantitative features (area, intensity)

This work was completed as part of the **INF7093 – Advanced Image Analysis** course at UQO.

---

## 🧠 Methodology

The pipeline consists of:

### 1. Preprocessing
- RGB → Grayscale conversion
- Gaussian filtering
- Histogram equalization
- Otsu thresholding
- Morphological operations

### 2. Segmentation
- Distance transform
- Local maxima detection
- Watershed algorithm

### 3. Analysis
- Cell counting
- Area calculation
- RGB intensity analysis

👉 The Watershed method successfully separates touching cells, unlike region-based or K-means methods.

---

## 📊 Results

- ✅ 21 cells detected
- Accurate separation of overlapping cells
- Reliable morphological and photometric analysis

---

## 📁 Project Structure
