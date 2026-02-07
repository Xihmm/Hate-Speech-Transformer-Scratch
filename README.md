# Encoder-Only Transformer from Scratch for Hate-Speech Detection

## 📌 Project Overview
This research project implements an **Encoder-Only Transformer architecture from scratch** using PyTorch to address text classification. We validated the implementation on the **HateXplain dataset**, achieving competitive results for a non-pretrained model.

## 🛠 Technical Implementation
* **Self-Attention**: Built Multi-head Attention (8 heads) to capture nuanced language patterns.
* **Architecture**: 6 stacked encoder layers with a model dimension of 256 and feedforward dimension of 512.
* **Training Strategy**: 
    * **Optimizer**: AdamW with weight decay.
    * **Scheduler**: OneCycleLR for faster convergence.
    * **Regularization**: 10% Dropout and Early Stopping based on validation loss.

## 📊 Performance Benchmark
The model achieved **52.3% accuracy** on the HateXplain test set without any pre-trained weights.

| Model | Accuracy (%) | Source |
| :--- | :---: | :--- |
| BERT (Pre-trained) | 69.0 | HateXplain Paper  |
| **Our Transformer (From Scratch)** | **52.3** | **This Work** |

## 📂 Project Structure
* `cust_transformer.py`: Core logic for self-attention and encoder blocks.
* `train.py`: Training loop with OneCycleLR and Weights & Biases (W&B) integration.
* `Structure diagram.png`: Structure diagram.


## 👥 Contributors
* **Ruby Gong & Kevin Lu** 
