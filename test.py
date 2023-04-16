import torch
from torch import nn
from config import Config
from data.load import TableDataLoader
from load import load
from model.cmha import CompressedMultiHeadAttention
import cProfile

if __name__ == '__main__':
    args = Config('./configs/ncgm.yaml')
    # tr_dl = TableDataLoader(args, "train")
    # table_train_dl = tr_dl.get_dataloader()
    # for batch in table_train_dl:
    #     torch.save(batch, 'batch.pkl')
    #     break
    model, criterion, optimizer = load(args)

    batch = torch.load('batch.pkl')
    geometry, appearance, content, bounding_box, row_adj_matrix, col_adj_matrix, cell_adj_matrix, structure = batch
    # bounding_box = bounding_box.squeeze(0)
    def f_b():
        cell_output, row_output, col_output, emb_pairs = model(geometry, appearance, content, bounding_box)
        # cProfile.run('model(geometry, appearance, content, bounding_box)')
        loss = criterion(cell_output, row_output, col_output, emb_pairs, cell_adj_matrix, row_adj_matrix, col_adj_matrix)
        loss.backward()
    
    cProfile.run('f_b()')