import torch
from torch import nn
import os
import json
from logger import logger
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
        self.args = args
        self.mode = mode

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path {self.path} not found")
        self.data_list = self.load_file_path()

    def load_file_path(self):
        """ Load file path

        此处加载三种数据
        1. chunk_path: chunk 文件中是单元格的坐标信息 [x1, y1, x2, y2]，以及单元格的文本信息
        2. img_path: 图像文件
        3. structure_path: structure 文件中是表格的结构信息，包括单元格的逻辑位置，start_row, start_col, end_row, end_col，是标签

        Returns:
            data (list): list of dict, each dict contains chunk_path, img_path, structure_path
        """
        # load chunk path
        data = []
        for file in os.listdir(os.path.join(self.path, self.mode, "chunk")):
            if file.endswith(".chunk"):
                file_name = file[: -len(".chunk")]
                data.append(
                    {
                        "chunk_path": os.path.join(self.path, self.mode, "chunk", file),
                        "img_path": os.path.join(
                            self.path, self.mode, "img", file_name + ".png"
                        ),
                        "structure_path": os.path.join(
                            self.path, self.mode, "structure", file_name + ".json"
                        ),
                    }
                )
        assert len(data) == len(os.listdir(os.path.join(self.path, self.mode, "chunk")))
        return data

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

        return img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        cp, ip, sp = item["chunk_path"], item["img_path"], item["structure_path"]
        for p in [cp, ip, sp]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Path {p} not found")
        with open(sp, "r") as f:
            structure = json.load(f)
        with open(cp, "r") as f:
            chunk = json.load(f)
        img = self.load_img(ip)
        return chunk, img, structure

