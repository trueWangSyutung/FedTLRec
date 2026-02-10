# FedTLRec: Federated Recommendation with Transformer-based Parameter Aggregation and LoRA Compression

## Project Overview

This project implements a federated learning-based recommender system, employing a combination of K-means clustering and CoRA (Collaborative LoRA) techniques to improve the performance and efficiency of the recommender system.

### Core Features

- **Federated Learning Architecture**

- **K-means Clustering**

- **CoRA Technology**

- **Personalized Recommendation**

## Project Structure

```bash

├── data.py # Data processing and loading module

├── engine.py # Federated learning engine, responsible for training and evaluation logic

├── fedmodel.py # Server-side model definition

├── mlp.py # Client-side MLP model and CoRA embedding implementation

├── utils.py # Collection of utility functions

├── metrics.py # Evaluation metric calculation

└── README.md # Project documentation

```

## Main Components

### 1. Data Processing ([data.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py))

- [UserItemRatingDataset](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py#L8-L25): User-item rating dataset wrapper

- [SampleGenerator](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py#L27-L153): Data preprocessing and negative sampling generator

- Supports both explicit and implicit feedback modes

### 2. Client-side model ([mlp.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py))

- [CoRACommonEmbedding](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L46-L108): Implements LoRA embedding layers, supporting rank adaptation and SVD initialization

- [MLP](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L111-L204): Multilayer perceptron recommendation model, with the option to use Transformer or KAN layers

- Supports multiple activation functions and network configurations

### 3. Server-side model ([fedmodel.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py))

- [ServiceModel](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py#L191-L236): Transformer-based server-side aggregation model

- [ServiceModelMLP](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py#L94-L150): Server-side aggregation model based on MLP (backup solution)

- Responsible for aggregating parameters from clients and generating global knowledge

### 4. Federated Learning Engine ([engine.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/engine.py))

- [Engine](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/engine.py#L15-L361): Core training engine, managing the federated learning process

- Supports two aggregation strategies: K-means clustering and ordinary grouping

- Implements differential privacy protection mechanism

- Includes complete training and evaluation loops

## Runtime Environment

- Python 3.9+

- PyTorch 1.10+

- Related dependencies: scikit-learn, numpy, pandas, tqdm, etc.

## Usage

1. Prepare the dataset (standard recommendation system dataset format such as MovieLens)

2. Configure model parameters (set in the main training script)

3. Run the training script:

```bash
python main.py --config config.json

```

## Configuration Parameter Description

Main configuration items include:
- [latent_dim](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L117-L117): Embedding dimension
- [r](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/sh_result/r.py#L0-L92): LoRA rank parameter
- [num_users](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L115-L115), [num_items](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L116-L116): Number of users and items
- `lr`: Learning rate
- `batch_size`: Batch size
- `local_epoch`: Number of local training epochs
- `clients_sample_ratio`: Client sampling ratio
- `use_kmean`: Whether K-means clustering is enabled
- `dp`: Differential privacy noise coefficient
- 
## Cite This Article
```bibtex
@article{Wang2026FedTLRec,
  author = {Xudong Wang and Ruixin Zhao},
  title = {FedTLRec: Federated Recommendation with Transformer-based Parameter Aggregation and LoRA Compression},
  journal = {ICCK Transactions on Machine Intelligence},
  year = {2026},
  volume = {2},
  number = {2},
  pages = {65-76},
  doi = {10.62762/TMI.2025.882476},
  url = {https://www.icck.org/article/abs/TMI.2025.882476},
  abstract = {Federated learning has emerged as a key paradigm in privacy-preserving computing due to its "data usable but not visible" property, enabling users to collaboratively train models without sharing raw data. Motivated by this, federated recommendation systems offer a promising architecture that balances user privacy with recommendation accuracy through distributed collaborative learning. However, existing federated recommendation systems face significant challenges in balancing model performance, communication efficiency, and user privacy. In this paper, we propose FedTLRec (Federated Recommendation with Transformer-based Parameter Aggregation and Collaborative LoRA), which introduces a federated recommendation framework that integrates Low-Rank Adaptation (LoRA) for parameter compression and Transformer-based aggregation. It addresses key challenges in communication efficiency and model performance by compressing client updates via LoRA and employing a Transformer model with attention mechanisms to effectively aggregate parameters from multiple clients. A K-means clustering strategy further enhances efficiency by grouping similar clients. Experiments on real-world datasets show that FedTLRec achieves superior recommendation accuracy with significantly reduced communication costs, while maintaining robust performance in client dropout scenarios. Code is available at: https://github.com/trueWangSyutung/FedTLRec.},
  keywords = {federated recommendation, low-rank adaptation, transformer},
  issn = {3068-7403},
  publisher = {Institute of Central Computation and Knowledge}
}
```



