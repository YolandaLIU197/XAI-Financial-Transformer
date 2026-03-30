# XAI Financial Transformer: Explainability in Regime-Switching Markets

This project explores explainable AI (XAI) techniques for financial prediction using deep learning models, with a focus on understanding how model behavior and feature importance change across different market regimes.

Full report: see `report.pdf`

---

## Overview

Machine learning models are increasingly used in financial decision-making, but their lack of interpretability limits trust and adoption.

This project addresses this challenge by combining:
- Regime-aware modeling  
- Gradient-based attribution methods  
- Attention-based interpretability  

to provide a more robust and economically meaningful explanation of model behavior.

---

## Project Pipeline

1. **Data Preprocessing & Expansion**
   - Expanded monthly financial data (~750 samples) into daily (~20,000 samples)
   - Constructed weekly and bi-weekly aggregated datasets to balance noise and sample size  
   - Based on Fama-French 5 factors and portfolio returns  

2. **Regime Segmentation (HMM)**
   - Applied Gaussian Hidden Markov Model (HMM) to detect market regimes  
   - Ensured temporal coherence and economic interpretability  
   - Used regimes for downstream model evaluation and explainability  

3. **Model & Explainability**
   - Transformer-based architecture with factor-as-token representation  
   - Gradient-based attribution methods:
     - Vanilla Gradient  
     - SmoothGrad  
     - Integrated Gradients  
   - Attention-based methods:
     - Embedding-level attention visualization  
     - Attention rollout across layers  

---

## Key Findings

- **Market factor (Mkt-RF) consistently dominates model behavior** across regimes  
- Raw attention is **unstable and insufficient** as a standalone explanation  
- **Attention rollout provides more stable and global interpretability** by capturing information flow across layers  
- Strong alignment between **gradient attribution and attention rollout** confirms key economic drivers  
- Model explanations vary across regimes, highlighting the importance of **regime-aware analysis**  

These findings demonstrate that explainability must be evaluated jointly with model structure and market conditions.

--

## Repository Structure
XAI-Financial-Transformer/
├── report.pdf
├── notebooks/
│ ├── 1_data_processing.ipynb
│ ├── 2_hmm_regime_segmentation.ipynb
│ ├── 3_attention_visualization.ipynb
