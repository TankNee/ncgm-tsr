import torch
from torch import nn

from config import Config
from model.cmha import CompressedMultiHeadAttention

class EgoContextExtractor(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(EgoContextExtractor, self).__init__(**kwargs)
        self.cmha = CompressedMultiHeadAttention(args)


    def build_graph(self, X: torch.tensor):
        """Build Graph

        formulation: $h_{\Theta}\left(\mathbf{x}_i, \mathbf{x}_j\right)=\mathbf{x}_i \|\left(\mathbf{x}_i-\mathbf{x}_j\right)$
        
        Args:
            X: input tensor, shape: (batch_size, num_nodes, num_hidden)
        """
        # kronecker product

        pass

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