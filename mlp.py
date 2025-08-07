import math

import torch
from engine import Engine
from fastkan import AttentionWithFastKANTransform, FastKANLayer
from utils import use_cuda, resume_checkpoint
import torch.nn.functional as F
from fedmodel import TransformerBlock

# 计算模型参数量
def compute_trainable_params_size(model):
    """
    计算模型中可训练参数的数量并转换为字节大小

    Args:
        model: PyTorch 模型

    Returns:
        int: 可训练参数的字节大小
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            param_size = param.numel() * param.element_size()
            total_params += param_size
    return total_params / (1024 ** 2)


def compute_trainable_params_count(model):
    """
    计算模型中可训练参数的数量（仅计数）

    Args:
        model: PyTorch 模型

    Returns:
        int: 可训练参数的数量
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    # 转换为字节大小

    return total_params

class CoRACommonEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, r: int = 8,
                 lora_alpha: int = 16, device: torch.device = torch.device('cpu'),full: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.device = device
        self.full = full
        self.embedding = torch.nn.Embedding(embedding_dim, output_dim)
        self.lora_A = torch.nn.Parameter(torch.zeros(r, output_dim))
        self.lora_B = torch.nn.Parameter(torch.zeros(embedding_dim, r))
        if not full:
            self.embedding.weight.requires_grad = False
        self.rank_pattern = torch.nn.Parameter(torch.ones(r))  # 重要性权重
        self.rank_threshold = 0.1  # 秩剪枝阈值
        self.scaling = self.lora_alpha / self.r  # 修改缩放因子计算方式
        # 参数初始化，lora_A 初始值 为 随机 正态分布，lora_B 初始值为 0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.dropout = torch.nn.Dropout(0.1)
        # 计算上传参数压缩率  lora_A / self.embedding.weight
        self.upload_press_rate = (r*self.output_dim)/(self.embedding_dim*self.output_dim)
        print(f"上传参数压缩率：{self.upload_press_rate:.4f}")
        print(f"模型参数数量：{compute_trainable_params_size(self):.2f}MB")

        # 只在非全训练模式下使用SVD初始化
        if not full:
            self._svd_init()

    def _svd_init(self):
        """SVD初始化策略，提高训练稳定性"""
        with torch.no_grad():
            # 创建随机权重矩阵进行SVD分解
            temp_weight = torch.randn(self.embedding_dim, self.output_dim) * 0.02
            U, S, Vh = torch.linalg.svd(temp_weight, full_matrices=False)

            # 使用SVD结果初始化LoRA参数
            rank = min(self.r, min(U.shape[1], Vh.shape[0]))
            self.lora_B.data[:, :rank] = U[:, :rank]
            self.lora_A.data[:rank, :] = torch.diag(S[:rank]) @ Vh[:rank, :]

    def get_effective_rank(self):
        """获取有效秩（用于AdaLoRA）"""
        # 添加缺失的adaptive_rank属性
        self.adaptive_rank = hasattr(self, 'adaptive_rank') and self.adaptive_rank
        if self.adaptive_rank:
            return torch.sum(self.rank_pattern > self.rank_threshold).item()
        return self.r

    def forward(self, x):
        # 获取基础权重
        important_ranks = self.rank_pattern > self.rank_threshold
        effective_lora_A = self.lora_A * important_ranks.unsqueeze(-1)  # [r, output_dim]
        lora_weight = torch.matmul(self.lora_B, effective_lora_A) * self.scaling

        # 应用dropout
        lora_weight = self.dropout(lora_weight)
        
        # 组合权重并前向传播
        combined_weight = self.embedding.weight + lora_weight
        return torch.nn.functional.embedding(x, combined_weight)


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = CoRACommonEmbedding(embedding_dim=self.num_items,
                                                  output_dim=self.latent_dim,
                                                  r=config["r"],
                                                  lora_alpha=config['lora_alpha'],
                                                  full=config["full_train"]
                                                  )
        if config['use_transfermer'] is True:
            self.attention_layers = torch.nn.ModuleList()
            for _ in range(config["transfermer_block_num"]):
                if config['use_kan_transfermer'] is True:
                    self.attention_layers.append(AttentionWithFastKANTransform(
                        self.latent_dim,self.latent_dim,self.latent_dim,
                        4,
                       8, gating=True
                    ))
                else:
                    self.attention_layers.append(TransformerBlock(
                        self.latent_dim, self.latent_dim,
                        dropout=0.1
                    ))


        self.fc_layers = torch.nn.ModuleList()
        if config['use_kan'] is True:
            for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
                self.fc_layers.append(FastKANLayer(
                    in_size, out_size,
                    grid_min=-1,
                    grid_max=1,
                    num_grids=2,
                    use_base_update=False,
                    base_activation=F.tanh,
                    spline_weight_init_scale=0.1,
                ))
            self.affine_output = FastKANLayer(
                    config['layers'][-1], 1,
                    grid_min=-1,
                    grid_max=1,
                    num_grids=2,
                    use_base_update=False,
                    base_activation=F.tanh,
                    spline_weight_init_scale=0.1,
                )
        else:
            for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        print("参数数量：", sum(p.numel() for p in self.parameters()))
        print("参数数量：", compute_trainable_params_size(self))
        total_params = compute_trainable_params_count(self)
        print(f"模型参数大小为：{total_params} 字节")
        print(self)

    def forward(self, item_indices):
        # 修复user_embedding生成方式，确保在GPU上正确运行
        batch_size = item_indices.size(0)
        user_indices = torch.zeros(batch_size, dtype=torch.long, device=item_indices.device)
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        if self.config['use_transfermer'] is True:
            item_embedding = self.attention_layers[0](user_embedding, item_embedding, item_embedding)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            if self.config['function']  == "relu":
                vector = torch.nn.ReLU()(vector)
            elif self.config['function']  == "leaky_relu":
                vector = torch.nn.LeakyReLU()(vector)
            elif self.config['function']  == "elu":
                vector = torch.nn.ELU()(vector)
            elif self.config['function']  == "gelu":
                vector = torch.nn.GELU()(vector)
            elif self.config['function']  == "tanh":
                vector = torch.nn.Tanh()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        # 添加数值稳定性处理
        rating = torch.clamp(rating, min=1e-6, max=1-1e-6)
        return rating

    def init_weight(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
