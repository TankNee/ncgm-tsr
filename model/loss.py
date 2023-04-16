import torch
from torch import nn
import torch.nn.functional as F
from config import Config


class ContrastiveLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        self.alpha_margin = args[f"{args_prefix}.alpha_margin"]

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """Forward

        formulation: $\mathcal{L}_{\text {con }}=\left\|\mathbf{e}_{(a)}-\mathbf{e}_{(b)}^{+}\right\|_2^2+\max \left\{0, \alpha-\left\|\mathbf{e}_{(a)}-\mathbf{e}_{(b)}^{-}\right\|_2^2\right\}$

        Args:
            X: shape: (batch_size, num_node * num_node, num_hidden * 3 * 2), embedding pairs
            y: shape: (batch_size, num_node, num_node), labels, adjacency matrix
        """
        if X.shape[2] % 2 != 0:
            raise ValueError("X.shape[2] must be even")
        X_split = X.view(X.shape[0], X.shape[1], 2, -1)
        num_node = int(X.shape[1] ** 0.5)
        if num_node**2 != X.shape[1]:
            raise ValueError("X.shape[1] must be a square number")
        loss = torch.zeros((X.shape[0], X.shape[1]), device=X.device)
        # 用torch中的函数实现下面这段代码的逻辑而不是for循环
        # for i in range(X_split.shape[1]):
        #     idx_a, idx_b = i // num_node, i % num_node
        #     adj = y[:, idx_a, idx_b]
        #     pos_batch_idx = adj == 1    # 正样本的索引
        #     neg_batch_idx = adj == 0    # 负样本的索引
        #     if pos_batch_idx.sum() != 0:
        #         loss[pos_batch_idx, i] = torch.pow(
        #             X_split[pos_batch_idx, i, 0, :] - X_split[pos_batch_idx, i, 1, :], 2
        #         ).sum(dim=-1)
        #     if neg_batch_idx.sum() != 0:
        #         neg_loss = self.alpha_margin - torch.pow(X_split[neg_batch_idx, i, 0, :] - X_split[neg_batch_idx, i, 1, :], 2).sum(dim=-1)
        #         # 将loss中小于0的值置为0
        #         neg_loss = torch.where(neg_loss > 0, neg_loss, torch.zeros_like(neg_loss))
        #         loss[neg_batch_idx, i] = neg_loss

        y = y.view(-1, 1) # shape: (batch_size * num_node * num_node, 1)
        X_split = X_split.view(-1, 2, X_split.shape[-1]) # shape: (batch_size * num_node * num_node, 2, num_hidden * 3 * 2)
        distance = torch.pow(X_split[:, 0, :] - X_split[:, 1, :], 2).sum(dim=-1) # shape: (batch_size * num_node * num_node, )
        # y和loss_part1的对应项相乘，得到正样本的loss
        loss_part1 = distance * y.squeeze(dim=-1)
        # 1-y和loss_part1的对应项相乘，得到负样本的loss
        loss_part2 = (1 - y.squeeze(dim=-1)) * distance
        # 将负样本的loss中小于0的值置为0
        loss_part2 = torch.where(loss_part2 > 0, loss_part2, torch.zeros_like(loss_part2))
        # 将正样本和负样本的loss相加
        loss = loss_part1 + loss_part2
        # 将loss的shape变为(batch_size, num_node * num_node)
        loss = loss.view(X.shape[0], X.shape[1])
        # 将loss的shape变为(batch_size, )
        loss = loss.sum(dim=-1)
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(ClassificationLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """Forward

        Args:
            X: shape: (batch_size, num_node * num_node, 2)
            y: shape: (batch_size, num_node, num_node) adjacency matrix
        """
        # 根据邻接矩阵构建新的标签
        y = y.view(y.shape[0], -1).to(torch.long)
        X = X.view(X.shape[0], 2, -1)
        # 使用交叉熵损失函数
        return F.cross_entropy(X, y)


class MixLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(MixLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        self.lambda1, self.lambda2 = (
            args[f"{args_prefix}.lambda1"],
            args[f"{args_prefix}.lambda2"],
        )
        self.con_loss = ContrastiveLoss(args)
        self.class_loss = ClassificationLoss(args)

    def forward(self, logits: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        """Forward

        Given a pair of collaborative graph embeddings and corresponding concatenated vector

        Args:
            logits: shape: (batch_size, num_node * num_node, 2)
            X: shape: (batch_size, num_node * num_node, num_hidden * 3 * 2)
            y: shape: (batch_size, num_node, num_node)
        """
        con_loss = self.con_loss(X, y)
        class_loss = self.class_loss(logits, y)
        return self.lambda1 * con_loss + self.lambda2 * class_loss


class NCGMLoss(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(NCGMLoss, self).__init__(**kwargs)
        args_prefix = f"{args['mode']}.model.loss"
        self.cell_loss = MixLoss(args)
        self.row_loss = MixLoss(args)
        self.col_loss = MixLoss(args)

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
            X: shape: (batch_size, num_node * num_node, num_hidden * 3 * 2)
            y: shape: (batch_size, num_node, num_node)
        """
        cell_loss = self.cell_loss(cell_logits, X, cell_y)
        row_loss = self.row_loss(row_logits, X, row_y)
        col_loss = self.col_loss(col_logits, X, col_y)
        loss = cell_loss + row_loss + col_loss
        loss = loss.mean()  # 要用mean吗？
        return loss
