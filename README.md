# Clinical Decision Support System (CDSS)

A Brain Tumor Classification project using MRI scans. This system leverages deep learning to predict tumor types and provides an easy-to-use interface for inference.

---
# STREAMLIT DEMO: https://clinical-decision-support-system.streamlit.app/
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)


---

## Overview
This project implements a deep learning-based solution to classify brain MRI scans into different tumor types, such as:
- Meningioma
- Glioma
- Pituitary
- No Tumor  

The model is trained using PyTorch and uses **Inception v3** as the base architecture.

---

## Features
- Preprocessing and augmentation of MRI images
- Training, validation, and testing pipelines
- Interactive visualization of predictions
- Streamlit web application for easy access

---

## Dataset
The dataset used is the **Brain Tumor MRI Dataset** from Kaggle:
- [Masoud Nickparvar - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Organized into **Training** and **Testing** folders with subfolders for each class.

---

## Model
- **Architecture:** Inception v3 (pretrained)
- **Input Size:** 299x299 RGB images
- **Output Classes:** 4 (meningioma, glioma, pituitary, no tumor)
- **File:** `models/final_model.pth`  
> The model is stored using **Git LFS** due to its large size (>100MB).

---
## Deployment
## Deployment

To deploy the project locally and test the web interface:

### 1. Download necessary files
- `api.py` — the backend API script  
- `test.html` — the frontend HTML file  
- `final_model.pth` — the trained model file, **managed via Git LFS** and located in the `models/` folder

> ⚠️ Note: `final_model.pth` is tracked using Git LFS because it is larger than 100 MB. If you clone the repository, you need Git LFS to fetch the actual model file.

### 2. Install Git LFS (if not already installed)
- Windows: [Git LFS installer](https://git-lfs.github.com/)  
- Mac: `brew install git-lfs`  
- Linux: `sudo apt install git-lfs`  

After installation, run:
```bash
git lfs install


## Installation

Clone the repository:
```bash
git clone https://github.com/probablyguru/Clinical-Decision-Support-System.git
cd Clinical-Decision-Support-System
