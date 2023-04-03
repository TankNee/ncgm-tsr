import torch
from torch import nn
import os

from config import Config
from model.loss import NCGMLoss
from model.ncgm import NCGM


def load(args: Config):
    args_prefix = f"{args['mode']}"
    ncgm_model = NCGM(args)
    criterion = NCGMLoss(args)
    optimizer = torch.optim.Adam(ncgm_model.parameters(), lr=args[f"{args_prefix}.lr"])
    return ncgm_model, criterion, optimizer
