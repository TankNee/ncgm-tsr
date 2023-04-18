import gensim
import torch
import os
import json
import numpy as np
from logger import logger
import pandas as pd
from config import Config
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SciTSRDataset(Dataset):
    def __init__(self, args: Config):
        mode = args["mode"]
        args_prefix = f"{mode}.dataset"
        self.name = args[f"{args_prefix}.name"]
        self.path = args[f"{args_prefix}.path"]
        self.num_content_length = args[f"{mode}.model.ncgm.num_content_length"]
        self.num_block_padding = args[f"{args_prefix}.num_block_padding"]
        self.img_shape = tuple(
            [int(x) for x in args[f"{args_prefix}.img_shape"].split(",")]
        )
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            args[f"{args_prefix}.word2vec.path"], binary=True
        )
        logger.debug(f"Word2vec loaded from {args[f'{args_prefix}.word2vec.path']}")
        self.args = args
        self.mode = mode

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path {self.path} not found")
        self.data_list = pd.read_csv(
            os.path.join(self.path, f"{self.mode}.csv")
        ).values.tolist()

    def get_transform(self, img_height, img_width):
        img_transforms = []

        # 预设的固定图像大小，取出短边和长边
        min_length = min(self.img_shape)
        max_length = max(self.img_shape)

        # 取出图像的短边和长边
        img_min_length = min(img_height, img_width)
        img_max_length = max(img_height, img_width)

        # 计算短边和长边的缩放比例
        img_scale = float(min_length) / float(img_min_length)

        if int(img_scale * img_max_length) > max_length:
            img_scale = float(max_length) / float(img_max_length)

        # 缩放图像
        scaled_img_height = int(img_scale * img_height)
        scaled_img_width = int(img_scale * img_width)

        padding_left, padding_top = (max_length - scaled_img_width) // 2, (
            max_length - scaled_img_height
        ) // 2
        padding_right, padding_bottom = (
            max_length - scaled_img_width - padding_left,
            max_length - scaled_img_height - padding_top,
        )

        # 创建transforms
        # Image.BICUBIC 图像双三次插值
        img_transforms.append(
            transforms.Resize((scaled_img_height, scaled_img_width), Image.BICUBIC)
        )
        # Image.PAD 填充图像
        img_transforms.append(
            transforms.Pad(
                (padding_left, padding_top, padding_right, padding_bottom),
                padding_mode="edge",
            )
        )
        # Image.ToTensor 将图像转换为tensor
        img_transforms.append(transforms.ToTensor())
        # Image.Normalize 对图像进行归一化
        img_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return (
            img_scale,
            padding_left,
            padding_top,
            transforms.Compose(img_transforms),
        )

    def load_img(self, path):
        """Load image and convert to tensor
        Args:
            path (str): path to image
        Returns:
            torch.Tensor: image tensor with shape (C, H, W)
        """
        img = Image.open(path).convert("RGB")
        (
            img_scale,
            padding_left,
            padding_top,
            img_transform,
        ) = self.get_transform(img.height, img.width)
        img = img_transform(img)

        return img, img_scale, padding_left, padding_top

    def get_text_embedding(self, text):
        """Get word embedding of text
        Args:
            text (str): text
        Returns:
            list[str]: word embedding of text
        """
        text = text.lower()
        text = text.split(" ")
        text_embedding = []
        for word in text:
            if word in self.word2vec.key_to_index:
                text_embedding.append(self.word2vec[word])
            else:
                text_embedding.append(self.word2vec["unk"])
        if len(text_embedding) > self.num_content_length:
            text_embedding = text_embedding[: self.num_content_length]
        else:
            # 是直接用pad向量，还是专门置顶一个新的向量？
            text_embedding += [self.word2vec["pad"]] * (
                self.num_content_length - len(text_embedding)
            )

        return text_embedding

    def get_pad_block(self):
        """Get pad block
        Returns:
            dict: pad block
        """
        return {
            "pos": [0, 0, 0, 0],
            "text": "",
        }

    def get_adj_matrix(self, rel_path, structure):
        """Get adjacency matrix of table
        Args:
            rel_path (str): table cell relation file path
        Returns:
            torch.Tensor: adjacency matrix of table
        """
        row_adj_matrix = torch.zeros(self.num_block_padding, self.num_block_padding)
        col_adj_matrix = torch.zeros(self.num_block_padding, self.num_block_padding)
        # 两个单元格是否是一个合并单元格的子单元格
        cell_adj_matrix = torch.zeros(self.num_block_padding, self.num_block_padding)

        with open(os.path.join(self.path, self.mode, rel_path), "r") as f:
            rel = f.read().splitlines()
        rel = [r.split("\t") for r in rel]

        for cell in structure:
            if cell['id'] >= self.num_block_padding:
                continue
            for other in structure:
                if cell['id'] == other['id'] or other['id'] >= self.num_block_padding:
                    continue
                # 判断是否是同一行
                if cell['start_row'] <= other['start_row'] and cell['end_row'] >= other['end_row']:
                    row_adj_matrix[cell['id']][other['id']] = 1
                    row_adj_matrix[other['id']][cell['id']] = 1
                # 判断是否是同一列
                if cell['start_col'] <= other['start_col'] and cell['end_col'] >= other['end_col']:
                    col_adj_matrix[cell['id']][other['id']] = 1
                    col_adj_matrix[other['id']][cell['id']] = 1

        # 两个单元格是否是一个合并单元格的子单元格
        for i in range(min(len(structure), self.num_block_padding)):
            cell_adj_matrix[i, i] = 1

        return row_adj_matrix, col_adj_matrix, cell_adj_matrix

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """Get item by index

        Args:
            idx (int): index of item
        Returns:
            geometry (list): geometry of text segment bounding box. N * (x,y,w,h), 注意此处的x,y是相对于整个表格的，坐标轴的原点在左上角
            appearance (torch.Tensor): appearance of whole table image. N * (C,H,W)
            content (list): content of text segment bounding box.   N * (str)
            bounding_box (list): bounding box of text segment bounding box. N * (x1,y1,x2,y2)
            structure (list): structure label of table. N
            scale (float): scale of image. 1
        """
        item = self.data_list[idx]

        content = []  # content of text segment bounding box.
        geometry = []  # geometry of text segment bounding box. (x,y,w,h)
        bounding_box = []  # bounding box of text segment bounding box. (x1,y1,x2,y2)

        if self.mode == "train":
            _, chunk_path, _, image_path, rel_path, structure_path = item
        else:
            _, chunk_path, _, image_path, structure_path = item

        # whole table image
        appearance, scale, padding_left, padding_top = self.load_img(
            os.path.join(self.path, self.mode, image_path)
        )

        # text segment bounding box
        with open(os.path.join(self.path, self.mode, chunk_path), "r") as f:
            chunk = json.load(f)["chunks"]
        # padding chunk to a fixed length
        chunk = (
            chunk[: self.num_block_padding]
            if len(chunk) > self.num_block_padding
            else chunk + [self.get_pad_block()] * (self.num_block_padding - len(chunk))
        )
        for idx, cell in enumerate(chunk):
            x_min, x_max, y_min, y_max = cell["pos"]
            x_min = x_min * scale + padding_left
            y_min = y_min * scale + padding_top
            x_max = x_max * scale + padding_left
            y_max = y_max * scale + padding_top
            content.append(self.get_text_embedding(cell["text"]))
            geometry.append(
                [
                    (x_min + x_max) / 2,  # center x of bounding box
                    (y_min + y_max) / 2,  # center y of bounding box
                    x_max - x_min,
                    y_max - y_min,
                ]
            )
            bounding_box.append([x_min, y_min, x_max, y_max])

        # structure label
        with open(os.path.join(self.path, self.mode, structure_path), "r") as f:
            structure = json.load(f)["cells"]

        row_adj_matrix, col_adj_matrix, cell_adj_matrix = self.get_adj_matrix(
            rel_path, structure
        )
        # sort cell of structure by id
        structure = sorted(structure, key=lambda x: x["id"])
        # SciTSR 的评估方法 https://github.com/Academic-Hammer/SciTSR/blob/master/examples/eval.py
        # 这个数据集的标签是评估 Logical Relation

        # to tensor
        geometry = torch.tensor(geometry)
        content = torch.tensor(np.array(content))
        bounding_box = torch.tensor(bounding_box)

        return (
            geometry,
            appearance,
            content,
            bounding_box,
            row_adj_matrix,
            col_adj_matrix,
            cell_adj_matrix,
            structure,
        )

    @staticmethod
    def collate_fn(batch):
        """Collate batch data
        Args:
            batch (list): batch data
        Returns:
            geometry (torch.Tensor): geometry of text segment bounding box. N * (x,y,w,h)
            appearance (torch.Tensor): appearance of whole table image. N * (C,H,W)
            content (torch.Tensor): content of text segment bounding box.   N * (str)
            bounding_box (torch.Tensor): bounding box of text segment bounding box. N * (x1,y1,x2,y2)
            row_adj_matrix (torch.Tensor): row adjacency matrix. N * (num_node, num_node)
            col_adj_matrix (torch.Tensor): column adjacency matrix. N * (num_node, num_node)
            cell_adj_matrix (torch.Tensor): cell adjacency matrix. N * (num_node, num_node)
            structure (list): structure label of table. N
        """
        (
            geometry,
            appearance,
            content,
            bounding_box,
            row_adj_matrix,
            col_adj_matrix,
            cell_adj_matrix,
            structure,
        ) = zip(*batch)

        geometry = torch.stack(geometry, dim=0)
        appearance = torch.stack(appearance, dim=0)
        content = torch.stack(content, dim=0)
        # bounding_box = torch.stack(bounding_box, dim=0)
        row_adj_matrix = torch.stack(row_adj_matrix, dim=0)
        col_adj_matrix = torch.stack(col_adj_matrix, dim=0)
        cell_adj_matrix = torch.stack(cell_adj_matrix, dim=0)

        return (
            geometry,
            appearance,
            content,
            bounding_box,
            row_adj_matrix,
            col_adj_matrix,
            cell_adj_matrix,
            structure,
        )
