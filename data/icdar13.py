import torch
from torch import nn
import os
import json
from logger import logger
from config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ICDAR13Dataset(Dataset):
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

        Returns:
            data (list): list of dict, each dict contains chunk_path, img_path, structure_path
        """
        data = []
        with open(os.path.join(self.path, f"{self.mode}.txt"), "r") as f:
            for line in f.readlines():
                data.append(line.readlines().strip().split())
        return data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        pass
