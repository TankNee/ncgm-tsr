import gensim
from torch import nn
import os
import json
from logger import logger
import pandas as pd
from config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SciTSRDataset(Dataset):
    def __init__(self, args: Config):
        mode = args["mode"]
        args_prefix = f"{mode}.dataset"
        self.name = args[f"{args_prefix}.name"]
        self.path = args[f"{args_prefix}.path"]
        self.img_shape = tuple(
            [int(x) for x in args[f"{args_prefix}.img_shape"].split(",")]
        )
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            args[f"{args_prefix}.word2vec.path"]
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
            scaled_img_height,
            scaled_img_width,
            transforms.Compose(img_transforms),
        )

    def load_img(self, path):
        """Load image and convert to tensor
        Args:
            path (str): path to image
        Returns:
            torch.Tensor: image tensor with shape (H, W, C)
        """
        img = Image.open(path).convert("RGB")
        (
            img_scale,
            scaled_img_height,
            scaled_img_width,
            img_transform,
        ) = self.get_transform(img.height, img.width)
        img = img_transform(img)

        return img, img_scale

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """Get item by index

        Args:
            idx (int): index of item
        Returns:
            geometry (list): geometry of text segment bounding box. N * (x,y,w,h)
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
            _, chunk_path, _, image_path, _, structure_path = item
        else:
            _, chunk_path, _, image_path, structure_path = item
        # text segment bounding box
        with open(os.path.join(self.path, self.mode, chunk_path), "r") as f:
            chunk = json.load(f)["chunks"]
        for cell in chunk:
            x_min, x_max, y_min, y_max = cell["pos"]
            text = cell["text"]
            content.append(text)
            geometry.append(
                [
                    (x_min + x_max) / 2,  # center x of bounding box
                    (y_min + y_max) / 2,  # center y of bounding box
                    x_max - x_min,
                    y_max - y_min,
                ]
            )
            bounding_box.append([x_min, y_min, x_max, y_max])

        # whole table image
        appearance, scale = self.load_img(
            os.path.join(self.path, self.mode, image_path)
        )

        # structure label
        with open(os.path.join(self.path, self.mode, structure_path), "r") as f:
            structure = json.load(f)["cells"]

        # sort cell of structure by id
        structure = sorted(structure, key=lambda x: x["id"])
        # SciTSR 的评估方法 https://github.com/Academic-Hammer/SciTSR/blob/master/examples/eval.py
        # 这个数据集的标签是评估 Logical Relation

        return geometry, appearance, content, bounding_box, structure, scale
