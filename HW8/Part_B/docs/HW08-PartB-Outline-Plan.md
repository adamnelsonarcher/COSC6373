# HW08 Part B Outline and Plan

## Assignment Requirements (from prompt)

1. Train the autoencoder on **sunflower** images (split of choice) and show original vs reconstructed examples.
2. Evaluate anomalies for each other class (**daisy, dandelion, rose, tulip**) and compare MSE for normal vs anomalous images.
3. Output **density score distributions** for training, validation, and anomalies for each flower type (as in reference [2]).
4. Explain how autoencoders work for anomaly detection.
5. Describe other anomaly-detection methods besides autoencoders.

## Dataset Reference (Kaggle)

- Source: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?select=flowers
- Expected folder layout:
  - `flowers/sunflower/`
  - `flowers/daisy/`
  - `flowers/dandelion/`
  - `flowers/rose/`
  - `flowers/tulip/`

## Scope-Aligned Notebook Outline

1. **Setup + Data Access**
   - Imports (`tensorflow`, `numpy`, `matplotlib`, `sklearn.neighbors.KernelDensity`).
   - Download/load Kaggle dataset path.
   - Set `NORMAL_CLASS = "sunflower"`.

2. **Data Split (sunflower only for training)**
   - Build train/validation split from sunflower images only.
   - Keep held-out sunflower samples for normal test comparison.

3. **Convolutional Autoencoder**
   - Encoder: Conv2D + MaxPooling blocks.
   - Decoder: Conv2DTranspose blocks.
   - Loss: MSE.
   - Train using sunflower train/val.

4. **Required Reconstruction Output**
   - Show side-by-side original and reconstructed sunflower validation images.

5. **Anomaly Evaluation by Flower Type**
   - For each class in `{daisy, dandelion, rose, tulip}`:
     - Run reconstruction.
     - Compute per-image MSE.
   - Compare distributions/statistics with sunflower test MSE.

6. **Density Score Distributions (reference [2])**
   - Extract latent vectors from encoder.
   - Fit KDE on sunflower training latent vectors.
   - Compute log-density scores for:
     - sunflower train
     - sunflower validation
     - each anomaly class (daisy, dandelion, rose, tulip)
   - Plot score distributions (histograms) per class.

7. **Short Written Answers**
   - Q4: Explain anomaly logic (trained on normal only -> high error/low density for non-normal).
   - Q5: List alternatives (Isolation Forest, One-Class SVM, LOF, Deep SVDD, GAN-based, diffusion-based).

## Execution Plan (Time-Boxed)

- **Step 1 (10-15 min):** Reuse Part A autoencoder code, switch normal class to sunflower, run training.
- **Step 2 (10-15 min):** Add MSE evaluation for four anomaly classes and make comparison plots.
- **Step 3 (10-15 min):** Add latent KDE density scoring and distribution plots.
- **Step 4 (5-10 min):** Add concise written answers for Q4 and Q5.
- **Step 5 (5 min):** Export notebook to PDF and verify required outputs are visible.

## Deliverables Checklist

- `.ipynb` file for Part B.
- PDF report with:
  - reconstruction examples (sunflower),
  - MSE normal vs anomalies,
  - density score distributions,
  - Q4 and Q5 written responses.
