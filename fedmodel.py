import torch
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hid_dim: int, num_heads: int, dropout: float = 0.1,
                 use_cuda: bool = False, use_mps: bool = False):
        super().__init__()
        assert hid_dim % num_heads == 0, "hid_dim 必须能被 num_heads 整除"

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads

        # 线性变换层
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        if use_cuda:
            self


    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 线性变换
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(attention, dim=-1))

        # 应用注意力
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        return self.fc(x)


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1,
                 use_cuda: bool = False, use_mps: bool = False):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, 8, dropout, use_cuda, use_mps)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        if use_cuda:
            self



    def forward(self, query, key, value, mask=None):
        query = query
        key = key
        value = value
        if mask is not None:
            mask = mask

        attention = self.attention(query, key, value, mask).squeeze(1)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))

class ServiceModelMLP(nn.Module):
    def __init__(self, embedding_dim, output_dim,user_num,max_line, r):
        super(ServiceModel, self).__init__()
        self.hidden_units = embedding_dim
        self.r = r
        # 将 user_num 分割为 user_num/max_line + 1 个
        self.user_num = user_num
        self.max_line = max_line
        self.transfer_num = user_num // max_line + 1
        self.transfermers = nn.ModuleList([
            nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.hidden_units * self.r,
                        self.hidden_units * self.r*2,
                        bias=True,
                    ),
                    torch.nn.Linear(
                        self.hidden_units * self.r*2,
                        self.hidden_units * self.r,
                        bias=True,
                    )
                ]
            ) for _ in range(self.transfer_num)
        ])
        self.output_layer = nn.Linear(user_num, 1)

        # 打印参数量
        print("参数数量：", sum(p.numel() for p in self.parameters()))
        print(self)

    def forward(self, lora_A_widgets):
        # lora_A 形状 [r, output_dim]
        # 将 lora_A_widgets 扩充为 【transfer_num*max_line, r, output_dim]
        lora_A_widgets = torch.tensor(lora_A_widgets)
        nums = lora_A_widgets.shape[0]
        used_transfermerblock = min(nums // self.max_line+1, self.transfer_num)
        # 如果 nums < transfer_num*max_line
        if nums < used_transfermerblock * self.max_line:
            zeros = torch.zeros(used_transfermerblock * self.max_line - nums, self.r * self.hidden_units)
            lora_A_widgets = torch.cat([lora_A_widgets, zeros], dim=0)

        # 创建一个新的张量来存储结果，避免就地操作
        processed_widgets = []
        for i in range(used_transfermerblock):
            sub_item_emb = lora_A_widgets[i * self.max_line:(i + 1) * self.max_line]
            for layer in self.transfermers[i]:
                sub_item_emb = layer(sub_item_emb)
            processed_widgets.append(sub_item_emb)

        # 拼接所有处理后的子张量
        lora_A_widgets = torch.cat(processed_widgets, dim=0)


        # item_emb 形状也是 [r, output_dim]
        # 把拼接的 删除
        return lora_A_widgets[:nums]

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

# 服务器端 参数聚合模型 ， 是一个 Transfermer Encoder 模型
# 分别聚合来自训练集的参数 embedding_item.lora_A, 并且 给客户端参数一个离线率（0.1，离线的客户端参数为全零）
class ServiceModel(nn.Module):
    def __init__(self, embedding_dim, output_dim,user_num,max_line, r):
        super(ServiceModel, self).__init__()
        self.hidden_units = embedding_dim
        self.r = r
        # 将 user_num 分割为 user_num/max_line + 1 个
        self.user_num = user_num
        self.max_line = max_line
        self.transfer_num = user_num // max_line + 1
        self.transfermers = nn.ModuleList([
            TransformerBlock(
                self.hidden_units * self.r,
                self.hidden_units * self.r,
                dropout=0.0,
            ) for _ in range(self.transfer_num)
        ])

        # 打印参数量
        print("参数数量：", compute_trainable_params_count(self))
        print("参数大小：", compute_trainable_params_size(self) )
        print(self)

    def forward(self, lora_A_widgets):
        # lora_A 形状 [r, output_dim]
        # 将 lora_A_widgets 扩充为 【transfer_num*max_line, r, output_dim]
        lora_A_widgets = torch.tensor(lora_A_widgets)
        nums = lora_A_widgets.shape[0]
        used_transfermerblock = min(nums // self.max_line+1, self.transfer_num)
        # 如果 nums < transfer_num*max_line
        if nums < used_transfermerblock * self.max_line:
            zeros = torch.zeros(used_transfermerblock * self.max_line - nums, self.r * self.hidden_units)
            lora_A_widgets = torch.cat([lora_A_widgets, zeros], dim=0)

        # 创建一个新的张量来存储结果，避免就地操作
        processed_widgets = []
        for i in range(used_transfermerblock):
            sub_item_emb = lora_A_widgets[i * self.max_line:(i + 1) * self.max_line]
            processed_sub_item = self.transfermers[i](sub_item_emb, sub_item_emb, sub_item_emb)
            processed_widgets.append(processed_sub_item)

        # 拼接所有处理后的子张量
        lora_A_widgets = torch.cat(processed_widgets, dim=0)

        # item_emb 形状也是 [r, output_dim]
        # 把拼接的 删除
        return lora_A_widgets[:nums]
# ... existing code ...