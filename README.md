
# InsightGWAS: Transformer-Based Model for GWAS Analysis

## Overview
![Image text](c6e817ae02a7731e5f5bd141c218370.png)
InsightGWAS is a Transformer-based deep learning framework designed to prioritize genetic variants associated with complex diseases using **genome-wide association studies (GWAS) summary statistics** and **multi-modal genomic annotations**.

 **MDD-MA Transformer Model**: Pre-trained on Major Depressive Disorder (MDD) GWAS data and fine-tuned on Migraine (MA) data.

This project consists of four example： Python scripts for training, transfer learning,inference using a Transformer-based model and a baseline(without transfer) learning and inference .

## Required Libraries
Make sure you have the following Python libraries installed:
- torch
- pandas
- scikit-learn
- numpy

## Running the Scripts
Execute the scripts in the following order:

1. **Train the initial model on MDD data**:
   ```bash
   python 1_train_model.py
   ```

2. **Perform transfer learning using MA data**:
   ```bash
   python 2_transform_learning_model.py
   ```

3. **Run inference**:
   ```bash
   python 3_inference.py
   ```
After running the inference script, the results will be saved in a file named `Predicted_SNPs.txt`.
   
4. **Run baseline model(without transfer) learning and inference**:
   ```bash
   python 4_baseline_learning_inference.py
   ```
After running the 4_baseline_learning_inference, the results will be saved in a file named `Predicted_SNPs_Baseline.txt`.

## Note on Example Data
The example data used in these scripts is limited to SNPs from chromosome 6. This is intended for demonstration purposes, and the results do not represent the model's full performance as described in the research paper.


## Input Data
The input features for each SNP variant include **GWAS statistics** and **functional annotations**, structured as follows:

| Feature Category         | Description |
|--------------------------|-------------|
| **Genomic Position**     | `chr`, `bpos` (chromosome and base position) |
| **GWAS Summary Stats**   | `beta`, `se`, `p-value`, `sample size (N)` |
| **Regulatory Annotations** | `sQTL`, `eQTL`, `brain eQTL`, `all eQTL`, `mQTL` |
| **Chromatin Features**    | `OCRs (open chromatin regions)`, `encode`, `footprints` |
| **Transcription Factor Binding** | `tfbsConsSites`, `targetScanS.wgRna` |
| **Genomic Evolutionary Features** | `genomicSuperDups`, `CADD score`, `GWAVA score` |
| **LD-related metrics** | `ldscore`, `allele frequency (freq)` |
| **Previous GWAS Evidence** | `reported in previous GWAS` |

## Model Architecture

The Transformer model consists of:
- **Multi-Head Self-Attention**: Captures interactions between SNP features, allowing the model to learn complex regulatory effects.
- **Feed-Forward Neural Networks**: Introduces non-linearity to improve feature learning.
- **Positional Encoding**: Retains order information in feature representations.
- **Binary Classification Output**: Predicts whether an SNP is significantly associated with a disease (0/1 classification).

Each Transformer encoder consists of:
- **4 attention heads**
- **2 layers**
- **Hidden dimension of 64**
- **Fully connected output layer with sigmoid activation**

The model is trained using **binary cross-entropy loss (BCE Loss)** with the Adam optimizer.

## Data Preprocessing
The following preprocessing steps are applied to input data:
1. **Standardization**: GWAS summary statistics and functional annotation scores are standardized using `StandardScaler`.
2. **Train-Test Split**: Data is randomly split into **80% training and 20% validation**.
3. **Data Loading**: Tensor representations are created for efficient training using PyTorch's `DataLoader`.
4. **Class Balancing**: The dataset contains an imbalanced number of positive (disease-associated) and negative SNPs, so **weight adjustments** may be applied.
