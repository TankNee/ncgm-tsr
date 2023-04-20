import torch
from torch import nn
from config import Config
from data.load import TableDataLoader
from load import load
from model.cmha import CompressedMultiHeadAttention
import cProfile
from model.utils import writer

if __name__ == "__main__":
    args = Config("./configs/ncgm.yaml")
    # tr_dl = TableDataLoader(args, "train")
    # table_train_dl = tr_dl.get_dataloader()
    # for batch in table_train_dl:
    #     torch.save(batch, "batch.pkl")
    #     break
    model, criterion, optimizer = load(args)

    batch = torch.load("batch.pkl")
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

    # bounding_box = bounding_box.squeeze(0)
    def f_b():
        # cell_output, row_output, col_output, emb_pairs = model(
        #     geometry, appearance, content, bounding_box
        # )
        # cProfile.run('model(geometry, appearance, content, bounding_box)')
        # loss = criterion(cell_output, row_output, col_output, emb_pairs, cell_adj_matrix, row_adj_matrix, col_adj_matrix)
        # loss.backward()
        # writer.add_graph(model, (geometry, appearance, content, bounding_box))
        # 绘制邻接矩阵的图片，写入tensorboard
        # 将矩阵转为图片格式 CHW 3x512x512
        # row_adj_matrix = row_adj_matrix[0].cpu().detach().numpy().reshape(1, 512, 512)
        # col_adj_matrix = col_adj_matrix[0].cpu().detach().numpy().reshape(1, 512, 512)
        # cell_adj_matrix = cell_adj_matrix[0].cpu().detach().numpy().reshape(1, 512, 512)
        # writer.add_image(
        #     "row_adj_matrix",
        #     row_adj_matrix[0].cpu().detach().numpy().reshape(1, 500, 500),
        #     0,
        # )
        # writer.add_image(
        #     "col_adj_matrix",
        #     col_adj_matrix[0].cpu().detach().numpy().reshape(1, 500, 500),
        #     0,
        # )
        # writer.add_image(
        #     "cell_adj_matrix",
        #     cell_adj_matrix[0].cpu().detach().numpy().reshape(1, 500, 500),
        #     0,
        # )

        writer.add_image_with_boxes("table image", appearance[0], bounding_box[0])

    f_b()
