# Deep Learning Practice

A hands-on collection of notebooks and scripts exploring core deep learning concepts — from optimization fundamentals and convolutional networks to recurrent models, attention mechanisms, and transformer-based NLP.

Built primarily with **TensorFlow/Keras**, with supplementary work in **PyTorch** and **Hugging Face Transformers**.

---

## Overview

This repository documents a structured learning path through modern deep learning. Each exercise focuses on a single concept: implementing it from first principles where possible, training small models on standard benchmarks, and visualizing what the network learns.

**Datasets used:** MNIST (image classification), IMDB (sentiment / sequence modeling), synthetic time-series sequences, and custom image samples.

---

## Algorithms & Techniques

### Neural Network Foundations

| Topic | Description |
|-------|-------------|
| **Logistic Regression** | Binary classifier on padded IMDB sequences using a single sigmoid output layer |
| **Dense (Fully Connected) Networks** | Multi-layer perceptrons for classification and regression |
| **Activation Functions** | ReLU, sigmoid, and softmax in classification and regression heads |
| **Loss Functions** | Binary cross-entropy, sparse categorical cross-entropy, mean squared error (MSE) |

### Optimization

| Algorithm | Notes |
|-----------|-------|
| **SGD** | Stochastic gradient descent with configurable learning rate |
| **SGD + Momentum** | Momentum coefficient of 0.9 for faster convergence on IMDB |
| **Adam** | Adaptive learning rate; default optimizer across most experiments |
| **RMSprop** | Root mean square propagation for classification and regression |
| **Nadam** | Nesterov-accelerated Adam variant |

Compared side-by-side on MNIST classification (`1_optimizers_classifi.ipynb`) and regression tasks (`1_optimizer_regression.ipynb`).

### Weight Initialization

| Strategy | Purpose |
|----------|---------|
| **Uniform / Normal** | Baseline random initialization; compared on MNIST |
| **Zeros** | Demonstrates failure mode of symmetric initialization |
| **Glorot Uniform (Xavier)** | Variance scaling for sigmoid/tanh activations |
| **He Normal** | Scaling suited for ReLU activations |
| **Variance Scaling** | Generalized initializer (fan-in / fan-out, normal distribution) |

### Convolutional Neural Networks (CNNs)

| Concept | Implementation |
|---------|----------------|
| **2D Convolution (`Conv2D`)** | Manual NumPy convolution and Keras `Conv2D` layers |
| **Pooling (`MaxPooling2D`)** | Spatial downsampling in MNIST classifiers |
| **Padding & Stride** | `same` vs `valid` padding; stride-1 vs stride-2 effects on output shape |
| **Feature Maps** | Intermediate activations extracted via Keras functional API |
| **Filter Visualization** | Learned kernel weights plotted after training |
| **Image Filtering** | Edge detection, blur, and sharpen via convolution kernels (OpenCV / NumPy) |
| **Data Augmentation** | Rotation and zoom via `ImageDataGenerator` on MNIST |
| **Transferable Features** | Early-layer feature extraction from a trained CNN |

Architectures range from single-convolution demos to multi-layer CNNs on full MNIST (`3_cnn_on_1_img.ipynb`, `4_cnn_on_img_dtst.ipynb`).

### Recurrent Neural Networks (RNNs)

| Model | Application |
|-------|-------------|
| **LSTM** | Sequence prediction on synthetic time series; hidden state and `return_sequences` exploration |
| **GRU** | Gated Recurrent Unit compared against LSTM on identical forecasting tasks |
| **Variable Sequence Length** | Effect of window size (3 vs 5 timesteps) on LSTM performance |
| **Temporal Dependencies** | LSTM output shapes and per-timestep hidden states |
| **Time-Series Forecasting** | Next-value prediction with train/test splits (`5_lstm_gru_time_srs.ipynb`) |

### Sequence-to-Sequence & Attention

| Architecture | Description |
|--------------|-------------|
| **Encoder–Decoder** | Conceptual mapping from input tokens to latent state and back to output |
| **Seq2Seq (LSTM)** | LSTM encoder with sequence-to-sequence framing on IMDB embeddings |
| **Attention Mechanism** | Keras `Attention` layer over LSTM encoder/decoder outputs |
| **Seq2Seq + Attention** | End-to-end model combining embedding, dual LSTMs, attention, and classification head |
| **Attention Visualization** | Heatmaps of encoder–decoder and BERT self-attention weights |

### Transformers & NLP

| Model / Task | Details |
|--------------|---------|
| **BERT (`bert-base-uncased`)** | Self-attention weights extracted and visualized across tokens (PyTorch + Hugging Face) |
| **Named Entity Recognition (NER)** | Pretrained transformer pipeline for entity tagging (PERSON, ORG, LOC) |
| **Word Embeddings** | `Embedding` layer (vocabulary 5000) for IMDB sequence models |
| **BLEU Score** | Precision-based n-gram overlap metric for generated vs. reference text |

---

## Repository Structure

```
Dl_prac/
├── 1_optimizer_regression.ipynb      # Optimizers on regression (SGD, Adam, RMSprop)
├── 1_optimizers_classifi.ipynb       # Optimizers on MNIST classification
├── 2_weight_initi.ipynb              # Weight initialization strategies
├── 3_cnn_on_1_img.ipynb              # Single-image CNN feature extraction
├── 3_logistic_reg_sgd_momen.py       # Logistic regression + SGD momentum (IMDB)
├── 4_cnn_on_img_dtst.ipynb           # Multi-layer CNN on MNIST
├── 5_lstm_gru_time_srs.ipynb         # LSTM vs GRU time-series forecasting
├── 6_Seq2Seq_base.ipynb              # Seq2Seq and attention on IMDB
├── 7_BERT_NER.py                     # NER with Hugging Face transformers
├── 8_atteniton_vizualize.py          # BERT attention weight visualization
├── 16_uniform_v_normal_initi.py      # Uniform vs normal initialization
├── 20_varience_scaling_initi.py      # Variance scaling (He-style) initialization
├── 21_edge_detect_img.py             # Edge detection via convolution
├── 22_23_blur_sharpen.py             # Blur and sharpen filters
├── 25_manual_convl.py                # Manual 2D convolution (NumPy)
├── 26_visualize_convl.py             # Multi-kernel convolution visualization
├── 28_padding_stride.py              # Padding and stride experiments
├── 29_vizulaize_filters_cnn.py         # Visualize learned CNN filters
├── 30_extrct_feature_concl.py        # CNN feature map extraction
├── 35_cnn_data_augm.py               # CNN + data augmentation (MNIST)
├── 44_lstm_diff_seq_len.py           # LSTM with varying sequence lengths
├── 45_hidden_state_lstm.py           # LSTM hidden state inspection
├── 46_lstm_gru_np_arr_seq.py         # LSTM vs GRU benchmark
├── 50_lstm_temporal_dependencies.py  # Temporal dependency analysis
├── 54_attention_w_visualize.py       # Attention weight heatmaps
├── 55_seq2seq_w_attn.py              # Seq2Seq model with attention
├── 56_BLEU_score.py                  # BLEU score computation
├── 57_encoder_decoder.py             # Encoder–decoder concept demo
└── image.png                           # Sample image for filtering exercises
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A virtual environment (recommended)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/Dl_prac.git
cd Dl_prac

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install tensorflow torch transformers matplotlib opencv-python numpy jupyter
```

### Running

- **Notebooks:** `jupyter notebook` or open `.ipynb` files in VS Code / Cursor.
- **Scripts:** `python <filename>.py` from the project root.

> MNIST and IMDB datasets are downloaded automatically by TensorFlow on first use.

---

## Tech Stack

| Library | Role |
|---------|------|
| TensorFlow / Keras | Primary framework for CNNs, RNNs, optimizers, and attention layers |
| PyTorch | BERT model loading and attention extraction |
| Hugging Face Transformers | Pretrained BERT and NER pipeline |
| OpenCV | Image I/O and filter operations |
| Matplotlib | Feature map, filter, and attention visualizations |
| NumPy | Manual convolution and array operations |

---

## License

This project is for educational purposes. Feel free to use and adapt the code for learning.
