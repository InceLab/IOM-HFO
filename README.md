# Intraoperative-HFO-analysis
Cascaded Residual-based Dictionary Learning to Distinguish HFOs from Pseudo-HFOs

# Demo Code for the Paper:  
**"Using High-Frequency Oscillations from Brief Intraoperative Neural Recordings to Predict the Seizure Onset Zone"**

## Overview
This repository contains the demo code that implements **Cascaded Residual-Based Dictionary Learning (CRDL)**, feature extraction methods, and a pre-trained **Random Forest** model to classify real versus pseudo-HFOs from intraoperative neural recordings.

The demo includes the following components:
- Raw intraoperative iEEG data
- Pre-learned dictionaries
- Processed features
- Pre-trained classifier

### Contents:
1. [Demo Code Overview](#demo-code-overview)
2. [Requirements](#requirements)
3. [Running the Demo](#running-the-demo)
4. [Parameter Tuning](#parameter-tuning)
5. [Contact Information](#contact-information)

---

## Demo Code Overview

### 1. Pre-learned Dictionary
- **File**: `Dictionary_CRDL_ASLR.mat`
- A dictionary learned using **Cascaded Residual-Based Dictionary Learning** with **kSVD** and **ASLR**.

### 2. Extracted Features
- **File**: `ExtractedFeature_CRDL_ASLR.mat`
- Features extracted from labeled events, used for training the classifier.

### 3. Random Forest Model
- **File**: `RF_CRDL_ASLR_Model.mat`
- A pre-trained **Random Forest** model for distinguishing between real and pseudo-HFOs.

### 4. Raw iEEG Data
- **File**: `Demo_rawiEEG.mat` (email authors or download from [DABI](https://dabi.loni.usc.edu/dsi/R01NS112497/AI413RR27VOG))
- Multi-channel iEEG recorded at 2kHz (included in `data.data`).

### 5. Labeled HFO Events
- **File**: `TrainData_HFO_Event.mat` (email authors)
- Labeled HFO events from intraoperative monitoring (IOM) and epilepsy monitoring unit (EMU) recordings.

### 6. Labeled Real/Pseudo HFO Events
- **File**: `Labeled_Event.mat` (email authors)
- Real and pseudo-HFO events labeled for training the Random Forest model.

> **Note:** To obtain these files, please contact the authors:
> - Behrang Fazli Besheli: [fazlibesheli.behrang@mayo.edu](mailto:fazlibesheli.behrang@mayo.edu)
> - Nuri Firat Ince: [ince.nuri@mayo.edu](mailto:ince.nuri@mayo.edu)

---

## Requirements
To run the demo, ensure the following resources are available:

### External Toolboxes:
- **kSVD and OMP Functionality**:  
  Download OMP.m and kSVD.m from [this repository](https://github.com/Xiaoyang233/KSVD/tree/master).
- **Random Forest Toolbox**:  
  Download the toolbox from [this GitHub repo](https://github.com/erogol/Random_Forests).

### Required Data:
- Raw iEEG data (`Demo_rawiEEG.mat` or other from [DABI](https://dabi.loni.usc.edu/dsi/R01NS112497/AI413RR27VOG)).
- Labeled HFO events (`TrainData_HFO_Event.mat`).
- Labeled real/pseudo-HFO events (`Labeled_Event.mat`).

---

## Running the Demo

### 1. **DemoCode_CRDL**:
Implements **Cascaded Residual Dictionary Learning (CRDL)** with **ASLR** and **kSVD**.
- Requirement: Annotated HFO events (`TrainData_HFO_Event.mat`).

### 2. **Demo_FeatureExtraction**:
Extracts features quantifying the representation quality of candidate HFO events.
- Features include:
  - Global L2 approximation error
  - Variability of residuals
  - Maximum coefficient of representation
  - Temporal Eigenvalue in the coefficient matrix, etc.
- Requirement: Annotated events (`Labeled_Event.mat`).

### 3. **Demo_RF_Model_LOSOCV**:
Trains and tests a Random Forest model using leave-one-subject-out cross-validation (LOSO-CV).
- Requirement: Install the **Random Forest** toolbox.

### 4. **Demo_Run**:
The full pipeline, including:
- Amplitude-based HFO detection (Dual Band HFO detector).
- Pseudo-HFO elimination using the pre-learned dictionary and Random Forest classifier.
- Distinguishing real vs. pseudo-HFOs from raw iEEG.
- Requirement: Install the **Random Forest** and **kSVD** toolboxes and raw iEEG data.
  
---

## Parameter Tuning

### kSVD Parameters:
- **L**: 2 (Number of atoms)
- **K**: 48 (Dictionary elements)
- **Iterations**: 10
- **Initialization**: 'GivenMatrix' (random normalized data)

### ASLR Parameters:
- **Max Shift**: 0 (circular shift)
- **Buffer Size**: 128
- **Overlap**: 120 (moving window overlap)
- **Atoms**: 6 (number of atoms for recosntruction)
- **Smoothing Factor**: 6 

---

## Contact Information
For any questions or inquiries, feel free to contact the authors:
- **Behrang Fazli Besheli**: [fazlibesheli.behrang@mayo.edu](mailto:fazlibesheli.behrang@mayo.edu)
- **Nuri Firat Ince**: [ince.nuri@mayo.edu](mailto:ince.nuri@mayo.edu)
