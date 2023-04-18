import torch
from data.load import TableDataLoader
from load import load
from logger import logger
from config import Config
from tqdm import tqdm

config_path = './configs/ncgm.yaml'

@logger.catch
def train(args: Config):
    mode = args['mode']
    logger.info(f"Loading model config from {config_path}")
    model, criterion, optimizer = load(args)
    gpu_id = args[f"{mode}.gpu"]
    device = torch.device("cuda", gpu_id) if gpu_id >= 0 else torch.device("cpu")
    model.to(device)
    # optimizer.to(device)
    # criterion.to(device)
    logger.info(f"Model loaded to {device}")

    try:
        logger.info("Training started")
        tr_dl = TableDataLoader(args, "train")
        table_train_dl = tr_dl.get_dataloader()
        dl_bar = tqdm(table_train_dl)
        pbar = tqdm(range(args[f"{mode}.epoch"]))
        pbar.set_description("Epoch")
        for epoch in pbar:
            model.train()
            pbar.set_description(f"Epoch {epoch}")
            for batch in dl_bar:
                geometry, appearance, content, bounding_box, row_adj_matrix, col_adj_matrix, cell_adj_matrix, structure = batch
                geometry = geometry.to(device)
                appearance = appearance.to(device)
                content = content.to(device)
                bounding_box = [b.to(device) for b in bounding_box]
                row_adj_matrix = row_adj_matrix.to(device)
                col_adj_matrix = col_adj_matrix.to(device)
                cell_adj_matrix = cell_adj_matrix.to(device)

                cell_output, row_output, col_output, emb_pairs = model(geometry, appearance, content, bounding_box)
                loss = criterion(cell_output, row_output, col_output, emb_pairs, cell_adj_matrix, row_adj_matrix, col_adj_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                dl_bar.set_description(f"Loss: {loss.item()}")

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
    train(args)
    # test(args)