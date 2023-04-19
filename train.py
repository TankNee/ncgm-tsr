import torch
import torch.nn.functional as F
from data.load import TableDataLoader
from load import load
from logger import logger
from config import Config
from tqdm import tqdm
from model.utils import writer
from transformers import get_linear_schedule_with_warmup

config_path = "./configs/ncgm.yaml"


@logger.catch
def train(args: Config):
    mode = args["mode"]
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

        warmup_step = (
            args[f"{mode}.warmup_proportion"]
            * args[f"{mode}.epoch"]
            * len(table_train_dl)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=args[f"{mode}.epoch"] * len(table_train_dl),
        )

        pbar = tqdm(range(args[f"{mode}.epoch"]))
        pbar.set_description("Epoch")
        dl_bar = tqdm(table_train_dl)
        for epoch in pbar:
            model.train()
            pbar.set_description(f"Epoch {epoch}")
            for batch in dl_bar:
                (
                    geometry,
                    appearance,
                    content,
                    bounding_box,
                    row_adj_matrix,
                    col_adj_matrix,
                    cell_adj_matrix,
                    structure,
                ) = batch
                geometry = geometry.to(device)
                appearance = appearance.to(device)
                content = content.to(device)
                bounding_box = [b.to(device) for b in bounding_box]
                row_adj_matrix = row_adj_matrix.to(device)
                col_adj_matrix = col_adj_matrix.to(device)
                cell_adj_matrix = cell_adj_matrix.to(device)

                cell_output, row_output, col_output, emb_pairs = model(
                    geometry, appearance, content, bounding_box
                )
                loss, loss_map = criterion(
                    cell_output,
                    row_output,
                    col_output,
                    emb_pairs,
                    cell_adj_matrix,
                    row_adj_matrix,
                    col_adj_matrix,
                )

                # write to tensorboard
                writer.add_scalars(f"epoch-{epoch}/loss", loss_map, dl_bar.n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                dl_bar.set_description(f"Loss: {loss.item()}")
            writer.flush()

        # 对最后一个batch的数据进行eval
        # model.eval()
        # with torch.no_grad():
        #     # cell_output, row_output, col_output 转换成Structure文件
        #     row_pred = F.softmax(row_output, dim=-1)
        #     col_pred = F.softmax(col_output, dim=-1)

        #     # construct adjacency matrix
        #     row_adj_matrix = torch.zeros((row_pred.shape[0], row_pred.shape[1], row_pred.shape[1]))
        #     col_adj_matrix = torch.zeros((col_pred.shape[0], col_pred.shape[1], col_pred.shape[1]))

        #     for i in range(row_pred.shape[0]):
        #         for j in range(row_pred.shape[1]):
        #             row_adj_matrix[i][j][j] = row_pred[i][j][0]
        #             row_adj_matrix[i][j][j+1] = row_pred[i][j][1]
        #         row_adj_matrix[i][row_pred.shape[1]-1][row_pred.shape[1]-1] = 1
        #     pass


        logger.info("Training finished")
        writer.close()
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


if __name__ == "__main__":
    args = Config(config_path)
    train(args)
    # test(args)
