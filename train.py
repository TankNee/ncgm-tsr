import torch
from data import TableDataLoader
from load import load
from logger import logger
from config import Config
from tqdm import tqdm

config_path = 'configs/ncgm.yaml'

@logger.catch
def train(args: Config):
    mode = args['mode']
    logger.info(f"Loading model config from {config_path}")
    model, criterion, optimizer = load(args)
    gpu_id = args[f"{mode}.gpu"]
    device = torch.device("cuda", gpu_id) if gpu_id >= 0 else torch.device("cpu")
    model.to(device)
    logger.info(f"Model loaded to {device}")

    try:
        logger.info("Training started")
        pbar = tqdm(range(args[f"{mode}.num_epochs"]))
        pbar.set_description("Epoch")
        for epoch in pbar:
            model.train()

    except Exception as e:
        logger.error("Training failed")
        logger.exception(e)
        raise e

@logger.catch
def test(args):
    tr_dl = TableDataLoader(args, "train")
    table_train_dl = tr_dl.get_dataloader()
    for batch in table_train_dl:
        print(batch)
        break

if __name__ == '__main__':
    args = Config(config_path)
    # train(args)
    test(args)