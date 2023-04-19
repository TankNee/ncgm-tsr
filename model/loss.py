import torch
from torch import nn
import torch.nn.functional as F
from config import Config


class ContrastiveLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        mode = args["mode"]
        self.alpha_margin = args[f"{args_prefix}.alpha_margin"]
        self.num_block_padding = args[f"{mode}.dataset.num_block_padding"]

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """Forward

        formulation: $\mathcal{L}_{\text {con }}=\left\|\mathbf{e}_{(a)}-\mathbf{e}_{(b)}^{+}\right\|_2^2+\max \left\{0, \alpha-\left\|\mathbf{e}_{(a)}-\mathbf{e}_{(b)}^{-}\right\|_2^2\right\}$

        Args:
            X: shape: (batch_size, num_node * (num_node - 1) / 2, num_hidden * 3 * 2), embedding pairs
            y: shape: (batch_size, num_node * (num_node - 1) / 2, num_node), labels, adjacency matrix
        """
        if X.shape[2] % 2 != 0:
            raise ValueError("X.shape[2] must be even")
        X_split = X.view(X.shape[0], X.shape[1], 2, -1)
        mask = torch.triu(torch.ones((self.num_block_padding, self.num_block_padding), dtype=torch.bool), diagonal=1)
        y = y[:, mask].view(-1)  # shape: (batch_size * num_node * (num_node - 1) / 2, 1)
        # shape: (batch_size * num_node * (num_node - 1) / 2, 2, num_hidden * 3 * 2)
        X_split = X_split.view(-1, 2, X_split.shape[-1])
        # distance = torch.pow(X_split[:, 0, :] - X_split[:, 1, :], 2).sum(dim=-1) # shape: (batch_size * num_node * (num_node - 1) / 2, )
        # 二范数的平方
        distance = F.pairwise_distance(X_split[:, 0, :], X_split[:, 1, :])

        loss = (y) * torch.pow(distance, 2) + (1 - y) * torch.clamp(
            self.alpha_margin - torch.pow(distance, 2), min=0.0
        )
        loss = loss.mean()
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(ClassificationLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        mode = args["mode"]
        self.num_block_padding = args[f"{mode}.dataset.num_block_padding"]

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """Forward

        Args:
            X: shape: (batch_size, num_node * (num_node - 1) / 2, 2)
            y: shape: (batch_size, num_node, num_node) adjacency matrix
        """
        # 根据邻接矩阵构建新的标签
        mask = torch.triu(torch.ones((self.num_block_padding, self.num_block_padding), dtype=torch.bool), diagonal=1)
        y = y[:, mask]
        y = y.view(y.shape[0], -1).to(torch.long)
        X = X.view(X.shape[0], 2, -1)
        # 使用交叉熵损失函数
        return F.cross_entropy(X, y)


class MixLoss(nn.Module):
    def __init__(self, args: Config, name: str, **kwargs):
        super(MixLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        self.lambda1, self.lambda2 = (
            args[f"{args_prefix}.lambda1"],
            args[f"{args_prefix}.lambda2"],
        )
        self.name = name
        self.con_loss = ContrastiveLoss(args)
        self.class_loss = ClassificationLoss(args)

    def forward(self, logits: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        """Forward

        Given a pair of collaborative graph embeddings and corresponding concatenated vector

        Args:
            logits: shape: (batch_size, num_node * (num_node - 1) / 2, 2)
            X: shape: (batch_size, num_node * (num_node - 1) / 2, num_hidden * 3 * 2)
            y: shape: (batch_size, num_node, num_node)
        """
        con_loss = self.con_loss(X, y)
        class_loss = self.class_loss(logits, y)
        loss_map = {
            f"{self.name}_con": con_loss.item(),
            f"{self.name}_class": class_loss.item(),
        }
        return self.lambda1 * con_loss + self.lambda2 * class_loss, loss_map


class NCGMLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(NCGMLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        self.cell_loss = MixLoss(args, 'cell')
        self.row_loss = MixLoss(args, 'row')
        self.col_loss = MixLoss(args, 'col')

    def forward(
        self,
        cell_logits: torch.Tensor,
        row_logits: torch.Tensor,
        col_logits: torch.Tensor,
        X: torch.Tensor,
        cell_y: torch.Tensor,
        row_y: torch.Tensor,
        col_y: torch.Tensor,
    ):
        """Forward

        Args:
            cell_logits: shape: (batch_size, num_node, num_node, 2)
            row_logits: shape: (batch_size, num_node, num_node, 2)
            col_logits: shape: (batch_size, num_node, num_node, 2)
            X: shape: (batch_size, num_node * (num_node - 1) / 2, num_hidden * 3 * 2)
            y: shape: (batch_size, num_node, num_node)
        """
        cell_loss, cell_loss_map = self.cell_loss(cell_logits, X, cell_y)
        row_loss, row_loss_map = self.row_loss(row_logits, X, row_y)
        col_loss, col_loss_map = self.col_loss(col_logits, X, col_y)
        loss = cell_loss + row_loss + col_loss
        loss = loss.mean()  # 要用mean吗？

        # merge loss map
        loss_map = {}
        loss_map.update(cell_loss_map)
        loss_map.update(row_loss_map)
        loss_map.update(col_loss_map)

        return loss, loss_map
