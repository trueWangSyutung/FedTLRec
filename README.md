# Kmean-Lora PFedRec:  K-means-based CoRA Federated Learning for Recommendation 

## 项目概述

本项目实现了一个基于联邦学习的推荐系统，采用K-means聚类和CoRA（Collaborative LoRA）技术相结合的方式，旨在提高推荐系统的性能和效率。

### 核心特性

- **联邦学习架构**
- **K-means聚类**
- **CoRA技术**
- **个性化推荐**

## 项目结构

```bash
├── data.py              # 数据处理和加载模块
├── engine.py            # 联邦学习引擎，负责训练和评估逻辑
├── fedmodel.py          # 服务端模型定义
├── mlp.py               # 客户端MLP模型和CoRA嵌入实现
├── utils.py             # 工具函数集合
├── metrics.py           # 评估指标计算
└── README.md            # 项目说明文档
```


## 主要组件

### 1. 数据处理 ([data.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py))
- [UserItemRatingDataset](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py#L8-L25): 用户-物品评分数据集包装器
- [SampleGenerator](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/data.py#L27-L153): 数据预处理和负采样生成器
- 支持显式反馈和隐式反馈两种模式

### 2. 客户端模型 ([mlp.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py))
- [CoRACommonEmbedding](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L46-L108): 实现LoRA嵌入层，支持秩自适应和SVD初始化
- [MLP](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L111-L204): 多层感知机推荐模型，可选择使用Transformer或KAN层
- 支持多种激活函数和网络配置

### 3. 服务端模型 ([fedmodel.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py))
- [ServiceModel](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py#L191-L236): 基于Transformer的服务端聚合模型
- [ServiceModelMLP](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/fedmodel.py#L94-L150): 基于MLP的服务端聚合模型（备用方案）
- 负责聚合来自客户端的参数并生成全局知识

### 4. 联邦学习引擎 ([engine.py](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/engine.py))
- [Engine](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/engine.py#L15-L361): 核心训练引擎，管理联邦学习流程
- 支持K-means聚类和普通分组两种聚合策略
- 实现差分隐私保护机制
- 包含完整的训练和评估循环

## 关键技术特点

### CoRA (Collaborative LoRA)
- 使用低秩矩阵分解技术减少模型参数传输量
- 通过LoRA适配器实现高效的个性化和全局知识融合
- 支持自适应秩调整以优化模型复杂度

### 联邦聚合策略
1. **K-means聚类聚合**: 使用K-means算法对用户进行聚类，同类用户间共享更多知识
2. **随机分组聚合**: 将用户随机分成若干组进行知识聚合
3. **全连接聚合**: 所有用户参与全局知识聚合

### 差分隐私保护
- 支持在客户端参数上传时添加拉普拉斯噪声
- 保护用户隐私信息不被泄露

## 运行环境

- Python 3.9+
- PyTorch 1.10+
- 相关依赖库：scikit-learn, numpy, pandas, tqdm等

## 使用方法

1. 准备数据集（MovieLens等标准推荐系统数据集格式）
2. 配置模型参数（在主训练脚本中设置）
3. 运行训练脚本：

```bash
python main.py --config config.json
```


## 配置参数说明

主要配置项包括：
- [latent_dim](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L117-L117): 嵌入维度
- [r](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/sh_result/r.py#L0-L92): LoRA秩参数
- [num_users](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L115-L115), [num_items](file:///Users/wxd2zrx/Desktop/desktop/GPFedRec/mlp.py#L116-L116): 用户和物品数量
- `lr`: 学习率
- `batch_size`: 批处理大小
- `local_epoch`: 本地训练轮数
- `clients_sample_ratio`: 客户端采样比例
- `use_kmean`: 是否启用K-means聚类
- `dp`: 差分隐私噪声系数

## 实验结果

项目包含评估指标监控和可视化功能：
- Hit Ratio (HR@10)
- Normalized Discounted Cumulative Gain (NDCG@10)
- 支持不同超参数（如r值）对模型性能影响的分析

## 贡献与扩展

该项目为进一步研究联邦推荐系统提供了良好基础，可扩展方向包括：
- 更先进的聚合算法
- 更高效的个性化技术
- 更强的隐私保护机制
- 更复杂的模型架构（如图神经网络）
