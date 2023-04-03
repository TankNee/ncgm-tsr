import torch
from torch import nn
import os
import json
from data.scitsr import SciTSRDataset
from logger import logger
from config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TableDataLoader(object):
    def __init__(self, args: Config, mode: str):
        args_prefix = f"{mode}.dataset"
        self.batch_size = args[f"{mode}.batch_size"]
        self.name = args[f"{args_prefix}.name"]
        self.args = args
        self.mode = mode

        self.dataset_map = {"SciTSR": SciTSRDataset}

    def get_dataloader(self):
        if self.name not in self.dataset_map:
            raise ValueError(f"Dataset {self.name} not found")
        dataset = self.dataset_map[self.name](self.args)
        logger.info(f"Dataset {self.name} loaded")
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.mode == "train"
        )
        return dataloader
