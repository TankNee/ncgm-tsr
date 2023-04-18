
from torch.utils.tensorboard import SummaryWriter
import os

if not os.path.exists('logs/tensorboard'):
    os.makedirs('logs/tensorboard')
writer = SummaryWriter('logs/tensorboard', comment='ncgm model loss')