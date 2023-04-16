from logger import logger
from torch import nn
from config import Config


class AddNorm(nn.Module):
    """Add Norm Layer
    残差连接与 Layer Normalization
    ref: https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html#id6

    Args:
        X: input tensor
        Y: input tensor
    Returns:
        output tensor
    """

    def __init__(self, args: Config, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        args_prefix = f'{args["mode"]}.model.addnorm'
        # norm_shape = tuple([int(x) for x in args[f"{args_prefix}.norm_shape"].split(',')])
        norm_shape = args[f"{args_prefix}.norm_shape"]
        self.dropout = nn.Dropout(args[f"{args_prefix}.dropout"])
        # noinspection PyTypeChecker
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Network
    ref: https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html#id5

    Args:
        X: input tensor
    Returns:
        output tensor
    """

    def __init__(self, args: Config, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        args_prefix = f'{args["mode"]}.model.ffn'
        num_input = args[f"{args_prefix}.num_input"]
        num_hidden = args[f"{args_prefix}.num_hidden"]
        num_output = args[f"{args_prefix}.num_output"]

        self.dense1 = nn.Linear(num_input, num_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hidden, num_output)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class CompressedMultiHeadAttention(nn.Module):
    def __init__(self, args: Config, **kwargs):
        super(CompressedMultiHeadAttention, self).__init__(**kwargs)
        args_prefix = f'{args["mode"]}.model.cmha'
        self.mc_eps = args[f"{args_prefix}.mc_epsilon"]
        self.mc_emb = int(args[f"{args_prefix}.num_hidden_emb"] / self.mc_eps)
        # multi head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=args[f"{args_prefix}.num_hidden_emb"],
            num_heads=args[f"{args_prefix}.num_heads"],
            batch_first=True,
            kdim=self.mc_emb,
            vdim=self.mc_emb,
        )
        # TODO: Memory Compression
        # element-wise addition and layer normalization
        self.addnorm1 = AddNorm(args)
        self.addnorm2 = AddNorm(args)
        self.mc_ln = nn.LayerNorm(self.mc_emb)

        # FFN: feed-forward network
        self.ffn = PositionWiseFFN(args)

    def memory_compress(self, X):
        # make X(M*N) to X(eM*N/e)

        X = X.reshape(X.shape[0], -1, self.mc_emb)
        # layer normalization
        X = self.mc_ln(X)
        return X

    def forward(self, query, key, value):
        key = self.memory_compress(key)
        value = self.memory_compress(value)
        # multi head attention
        output1, _ = self.mha(query, key, value)
        # element-wise addition and layer normalization
        output2 = self.addnorm1(query, output1)
        # FFN
        output3 = self.ffn(output2)
        # element-wise addition and layer normalization
        output = self.addnorm2(output2, output3)
        return output
