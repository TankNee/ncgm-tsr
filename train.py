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
    num_block_padding = args[f"{mode}.dataset.num_block_padding"]
    device = (
        torch.device("cuda", gpu_id)
        if gpu_id >= 0
        else torch.device("mps")
        if gpu_id == -2
        else torch.device("cpu")
    )
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
        for epoch in pbar:
            model.train()
            pbar.set_description(f"Epoch {epoch}")
            dl_bar = tqdm(table_train_dl)
            for batch in dl_bar:
                (
                    geometry,
                    appearance,
                    content,
                    bounding_box,
                    row_adj_matrix,
                    col_adj_matrix,
                    cell_adj_matrix,
                    chunks,
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
                writer.add_scalars(f"epoch_{epoch}/loss", loss_map, dl_bar.n)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args[f"{mode}.max_grad_norm"])
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                dl_bar.set_description(f"Loss: {loss.item()}")
            writer.flush()

        # 保存模型
        torch.save(model.state_dict(), args[f"{mode}.save_path"])
        logger.info(f"Model saved to {args[f'{mode}.save_path']}")

        # 对最后一个batch的数据进行eval
        model.eval()
        with torch.no_grad():
            # cell_output, row_output, col_output 转换成Structure文件
            row_pred = F.softmax(row_output, dim=-1)
            col_pred = F.softmax(col_output, dim=-1)

            # construct adjacency matrix
            row_adj_matrix_pred = torch.zeros(
                (row_pred.shape[0], num_block_padding, num_block_padding)
            )
            col_adj_matrix_pred = torch.zeros(
                (row_pred.shape[0], num_block_padding, num_block_padding)
            )
            mask = torch.triu(torch.ones((num_block_padding, num_block_padding), dtype=torch.bool), diagonal=1)
            row_adj_matrix_pred[:, mask] = row_pred
            col_adj_matrix_pred[:, mask] = col_pred

            row_adj_matrix_pred = row_adj_matrix_pred + torch.transpose(row_adj_matrix_pred, dim0=1, dim1=2)
            col_adj_matrix_pred = col_adj_matrix_pred + torch.transpose(col_adj_matrix_pred, dim0=1, dim1=2)

            writer.add_image_with_boxes("origin table", appearance[0], bounding_box[0])
            writer.add_image("adj_matrix/row/gt", row_adj_matrix[0])
            writer.add_image("adj_matrix/row/pred", row_adj_matrix_pred[0])
            writer.add_image("adj_matrix/col/gt", col_adj_matrix[0])
            writer.add_image("adj_matrix/col/pred", col_adj_matrix_pred[0])


            # construct chunks


        # construct structure
        # construct cell: start_row, start_col, end_row, end_col

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
