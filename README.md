# Wildfire Detection Project

## Overview
This project is part of the IT2011 Group Assignment for Artificial Intelligence and Machine Learning at SLIIT. The focus is on data cleaning, preprocessing, and Exploratory Data Analysis (EDA) for a wildfire detection dataset consisting of images classified as "fire" or "nofire." The goal is to prepare the dataset for machine learning models by applying various image preprocessing techniques, balancing classes, and generating visualizations to understand data distribution and quality.

The preprocessing pipeline includes resizing images, color balancing, normalization, denoising, edge detection for feature enhancement, and data augmentation to address class imbalance. EDA visualizations include histograms, bar plots for class distribution, and sample image comparisons before/after processing.

This repository contains individual contributions from group members (each handling a specific preprocessing technique), an integrated group pipeline, raw/processed data directories, and results.

## Dataset Details
- **Source**: The Wildfire Dataset from Kaggle [](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset).
- **Description**: The dataset includes approximately 10,000+ images divided into "fire" (images showing wildfires) and "nofire" (images without fire). It is pre-split into train (70%), validation (15%), and test (15%) sets.
  - Total images: ~7,000 train, ~1,500 val, ~1,500 test (exact counts may vary slightly after processing).
  - Image format: JPG, various resolutions (resized to 600x600 in preprocessing).
  - Classes: Binary classification ("fire" vs. "nofire").
  - Challenges: Class imbalance (more "nofire" images), varying lighting/contrast, noise from real-world captures.
- **Location in Repo**: Raw data is stored in `data/raw/`. Processed outputs are in `results/outputs/` and subdirectories for each technique (e.g., `results/color_balanced/`, `results/normalized/`, etc.).

## Group Members and Roles
Our group (Group ID: [Insert Group ID, e.g., Group_01]) consists of 6 members. Each member handled one preprocessing technique, implemented it in their individual notebook, and contributed to the group pipeline. Roles are as follows:

- **Member 1: IT24100260 - Resizing and Splitting**  
  Handled image resizing to a uniform 600x600 resolution and validated dataset splits. Notebook: `notebooks/IT_Number_Resizing_and_Splitting.ipynb`.

- **Member 2: IT24100356 - Color Conversion/Balancing**  
  Applied HSV color space conversion and histogram equalization on the Value channel to balance lighting/contrast. Notebook: `notebooks/IT_Number_Color_Conversion.ipynb`.

- **Member 3: IT24100368 - Normalization**  
  Normalized pixel values using ImageNet means/std for model compatibility. Notebook: `notebooks/IT_Number_Normalization.ipynb`.

- **Member 4: IT24100288 - Denoising**  
  Applied Gaussian blur and CLAHE enhancement to reduce noise and improve clarity. Notebook: `notebooks/IT_Number_Denoising.ipynb`.

- **Member 5: IT24100327 - Edge Detection**  
  Added a fourth channel with Sobel edge detection for feature enhancement. Notebook: `notebooks/IT_Number_Edge_Detection.ipynb`.

- **Member 6: IT24100304 - Data Augmentation**  
  Augmented images (rotations, flips, brightness adjustments) to balance classes in the training set. Notebook: `notebooks/IT_Number_Data_Augmentation.ipynb`.

All members collaborated on the integrated pipeline in `notebooks/group_pipeline.ipynb`, ensuring logical flow and commented code.

## How to Run the Code
### Requirements
- Python 3.8+ (tested on 3.12.3).
- Install dependencies via `pip install -r requirements.txt` (create this file if needed with the following):

torch
torchvision
opencv-python
albumentations
matplotlib
numpy
pillow
tqdm

- GPU recommended for faster processing (uses CUDA if available via PyTorch).
- Dataset: Download from the Kaggle link above and extract to `data/raw/` with subfolders `train/fire`, `train/nofire`, etc.

### Steps
1. **Setup Environment**: Clone this repo and install dependencies.
2. **Run Individual Notebooks**: For detailed steps and EDA:
 - Open Jupyter Notebook: `jupyter notebook`.
 - Run each `notebooks/IT_Number_[Technique].ipynb` in sequence (they build on prior outputs, e.g., resizing feeds into color balancing).
 - Each notebook includes code, justifications, outputs, and at least one EDA visualization (e.g., histograms for color channels, bar plots for class distribution).
3. **Run Group Pipeline**: Execute `notebooks/group_pipeline.ipynb` for the full end-to-end preprocessing flow. It integrates all techniques, processes the raw data, and saves final outputs to `results/outputs/`.
 - Input: `data/raw/`.
 - Output: Processed datasets in `results/` subdirs, EDA plots in `results/eda_visualizations/`.
4. **View Results**:
 - Visualizations: Check `results/eda_visualizations/` for PNG/JPEG files (e.g., class distribution bars, before/after image samples).
 - Logs: Optional execution logs in `results/logs/` (if generated).
 - Final Dataset: Processed images/features in `results/outputs/` (ready for model training).
5. **Notes**:
 - Paths in scripts are hardcoded (e.g., 'E:\\SLIIT\\...'); update to your local paths.
 - Processing time: ~10-30 minutes per technique on CPU; faster on GPU.
 - For reproduction: Ensure the raw dataset is in place before running.

This setup ensures reproducibility and demonstrates collaboration. For questions, contact the group lead at [insert email].