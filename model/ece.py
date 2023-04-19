import torch
from torch import nn

from config import Config
from model.cmha import CompressedMultiHeadAttention

class EgoContextExtractor(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(EgoContextExtractor, self).__init__(**kwargs)
        mode = args["mode"]
        num_hidden = args[f"{mode}.model.ncgm.num_hidden"]
        self.num_block_padding = args[f"{mode}.dataset.num_block_padding"]
        self.cmha = CompressedMultiHeadAttention(args)
        self.fc1 = nn.Linear(num_hidden, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.relu = nn.ReLU()

    def build_graph(self, X: torch.tensor):
        """Build Graph

        formulation: $h_{\Theta}\left(\mathbf{x}_i, \mathbf{x}_j\right)=\mathbf{x}_i \|\left(\mathbf{x}_i-\mathbf{x}_j\right)$
        
        Args:
            X: input tensor, shape: (batch_size, num_nodes, num_hidden)
        """
        # 取上三角矩阵
        mask = torch.triu(torch.ones((self.num_block_padding, self.num_block_padding), dtype=torch.bool), diagonal=1)        
        # xj - xi
        xj_minus_xi = (X.unsqueeze(2) - X.unsqueeze(1))[:, mask, :]
        xi = X.unsqueeze(1).repeat(1, X.shape[1], 1, 1)[:, mask, :]
        # fc(xj - xi) and fc(xi)
        xj_minus_xi = self.fc1(xj_minus_xi)
        xi = self.fc2(xi)
        edge_feat = self.relu(xj_minus_xi + xi) # shape: (batch_size, num_nodes * num_nodes, num_hidden)
        
        return edge_feat

    def forward(self, X: torch.tensor):
        """Forward

        Args:
            X: input tensor, shape: (batch_size, num_nodes, num_hidden) 第 l-1 层的 ECE 模块的输出
        """
        # build graph
        kv_embeddings = self.build_graph(X)
        # attention
        output = self.cmha(X, kv_embeddings, kv_embeddings)
        
        return output