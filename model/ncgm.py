import torch
from torch import nn
from config import Config
from torchvision import models as CVModels
from torchvision import ops as CVOps
from model.ece import EgoContextExtractor
from model.ccs import CrossContextSynthesizer


class CollaborativeBlock(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(CollaborativeBlock, self).__init__(**kwargs)
        self.geometry_ece = EgoContextExtractor(args)
        self.appearance_ece = EgoContextExtractor(args)
        self.content_ece = EgoContextExtractor(args)

        self.ccs = CrossContextSynthesizer(args)

    def forward(
        self,
        geometry_ece: torch.Tensor,
        appearance_ece: torch.Tensor,
        content_ece: torch.Tensor,
        geometry_ccs: torch.Tensor,
        appearance_ccs: torch.Tensor,
        content_ccs: torch.Tensor,
    ):
        """
        Args:
            geometry_ece: shape: (batch_size, num_nodes, num_hidden)
            appearance_ece: shape: (batch_size, num_nodes, num_hidden)
            content_ece: shape: (batch_size, num_nodes, num_hidden)
            geometry_ccs: shape: (batch_size, num_nodes, num_hidden)
            appearance_ccs: shape: (batch_size, num_nodes, num_hidden)
            content_ccs: shape: (batch_size, num_nodes, num_hidden)
        """
        geometry_ece_output = self.geometry_ece(geometry_ece)
        appearance_ece_output = self.appearance_ece(appearance_ece)
        content_ece_output = self.content_ece(content_ece)

        geometry_ccs_output, appearance_ccs_output, content_ccs_output = self.ccs(
            geometry_ccs,
            appearance_ccs,
            content_ccs,
            geometry_ece_output,
            appearance_ece_output,
            content_ece_output,
        )

        return (
            geometry_ece_output,
            appearance_ece_output,
            content_ece_output,
            geometry_ccs_output,
            appearance_ccs_output,
            content_ccs_output,
        )


class NCGM(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(NCGM, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.ncgm"
        num_hidden_concat = args[f"{args_prefix}.num_hidden"] * 3
        num_hidden_fc = args[f"{args_prefix}.num_hidden_fc"]
        num_backbone_filter_size = args[f"{args_prefix}.num_backbone_filter_size"]
        num_backbone_filter_out_channel = args[
            f"{args_prefix}.num_backbone_filter_out_channel"
        ]
        self.roi_align_size = args[f"{args_prefix}.roi_align_size"]
        self.num_block = args[f"{args_prefix}.num_block"]

        # Extract geometry features
        self.geometry_fc = nn.Sequential(
            nn.Linear(4, args[f"{args_prefix}.num_hidden"]),
            nn.ReLU(),
        )

        # Extract content features
        self.content_embedding = nn.Embedding(
            args[f"{args_prefix}.num_content_vocab"],
            args[f"{args_prefix}.num_hidden"],
        )


        # Extract appearance features
        self.resnet18 = CVModels.resnet18(pretrained=True)
        self.backbone_cnn = nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool,
            self.resnet18.layer1,
            nn.Conv2d(
                num_backbone_filter_out_channel,
                num_backbone_filter_out_channel,
                num_backbone_filter_size,
            ),
            nn.BatchNorm2d(num_backbone_filter_out_channel),
            nn.ReLU(),
            nn.Conv2d(
                num_backbone_filter_out_channel,
                num_backbone_filter_out_channel,
                num_backbone_filter_size,
            ),
            nn.BatchNorm2d(num_backbone_filter_out_channel),
            nn.ReLU(),
            nn.Conv2d(
                num_backbone_filter_out_channel,
                num_backbone_filter_out_channel,
                num_backbone_filter_size,
            ),
            nn.BatchNorm2d(num_backbone_filter_out_channel),
            nn.ReLU(),
        )
        # apply roi align in term of text bounding box
        self.appearance_fc = nn.Sequential(
            # 此处Linear的第一个参数是输入的维度，输入是roi_align的输出，该输出的形状是(num_node, num_feat_map, 2, 2)
            nn.Linear(
                self.roi_align_size**2 * num_backbone_filter_out_channel,
                args[f"{args_prefix}.num_hidden"],
            ),
            nn.ReLU(),
        )

        # collaborative blocks
        self.blocks = nn.Sequential()
        for idx in range(self.num_block):
            self.blocks.add_module(
                f"collaborative_block_{idx}", CollaborativeBlock(args)
            )

        # binary classification
        self.cell_fc = nn.Sequential(
            nn.Linear(num_hidden_concat, num_hidden_fc),
            nn.ReLU(),
            nn.Linear(num_hidden_fc, 2),
            nn.Softmax(dim=-1),
        )
        self.row_fc = nn.Sequential(
            nn.Linear(num_hidden_concat, num_hidden_fc),
            nn.ReLU(),
            nn.Linear(num_hidden_fc, 2),
            nn.Softmax(dim=-1),
        )
        self.col_fc = nn.Sequential(
            nn.Linear(num_hidden_concat, num_hidden_fc),
            nn.ReLU(),
            nn.Linear(num_hidden_fc, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, geometry, appearance, content):
        for block in self.blocks:
            geometry, appearance, content = block(geometry, appearance, content)

        # fused geometry, appearance, content
        emb = torch.cat([geometry, appearance, content], dim=-1)
        # generate pairs of embeddings
        # 整体重复 nodes 遍，单个 node 重复 nodes 遍，再将两个拼接，因为一共是 nodes * nodes 个，正好单个重复的和整体的数量一样，拼接起来就可以得到一个node和其他所有node的pair
        emb_pairs = torch.cat(
            [
                emb.unsqueeze(1).repeat(1, emb.shape[1], 1, 1),
                emb.unsqueeze(2).repeat(1, 1, emb.shape[1], 1),
            ],
            dim=-1,
        )
        # flatten, shape: (batch_size, num_nodes * num_nodes, num_hidden * 3)
        emb_pairs = emb_pairs.view(emb_pairs.shape[0], -1, emb_pairs.shape[-1])

        # binary classification
        cell_logits = self.cell_fc(emb_pairs)
        row_logits = self.row_fc(emb_pairs)
        col_logits = self.col_fc(emb_pairs)

        return cell_logits, row_logits, col_logits, emb_pairs
