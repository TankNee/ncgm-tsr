import torch
from torch import nn

from config import Config
from model.cmha import CompressedMultiHeadAttention


class CrossContextSynthesizer(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(CrossContextSynthesizer, self).__init__(**kwargs)
        # 3 parallel CMHA modules
        self.cmha1 = CompressedMultiHeadAttention(args)
        self.cmha2 = CompressedMultiHeadAttention(args)
        self.cmha3 = CompressedMultiHeadAttention(args)
        pass

    def forward(
        self,
        geometry: torch.Tensor,
        appearance: torch.Tensor,
        content: torch.Tensor,
        geometry_ece: torch.Tensor,
        appearance_ece: torch.Tensor,
        content_ece: torch.Tensor,
    ):
        """
        第 l 层的 CCS 模块

        Args:
            geometry: shape: (batch_size, num_nodes, num_hidden) 来自第 l-1 层的 CCS 模块
            appearance: shape: (batch_size, num_nodes, num_hidden) 来自第 l-1 层的 CCS 模块
            content: shape: (batch_size, num_nodes, num_hidden) 来自第 l-1 层的 CCS 模块
            geometry_ece: shape: (batch_size, num_nodes, num_hidden) 来自第 l 层的 ECE 模块
            appearance_ece: shape: (batch_size, num_nodes, num_hidden) 来自第 l 层的 ECE 模块
            content_ece: shape: (batch_size, num_nodes, num_hidden) 来自第 l 层的 ECE 模块

        Returns:
            geometry_output: shape: (batch_size, num_nodes, num_hidden)
            appearance_output: shape: (batch_size, num_nodes, num_hidden)
            content_output: shape: (batch_size, num_nodes, num_hidden)
        """
        geometry_output = self.cmha1(
            geometry,
            torch.cat([appearance_ece, content_ece], dim=1),
            torch.cat([appearance_ece, content_ece], dim=1),
        )
        appearance_output = self.cmha2(
            appearance,
            torch.cat([geometry_ece, content_ece], dim=1),
            torch.cat([geometry_ece, content_ece], dim=1),
        )
        content_output = self.cmha3(
            content,
            torch.cat([geometry_ece, appearance_ece], dim=1),
            torch.cat([geometry_ece, appearance_ece], dim=1),
        )

        return geometry_output, appearance_output, content_output
