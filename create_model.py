import torch

kernel_size = 3
dropout_percent = 0.5
embed_size = 128
n_res_blocks = 4
n_mlp_blocks = 4


class ResNetBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.activation = torch.nn.SiLU()
        self.batchnorm1 = torch.nn.BatchNorm2d(channels)
        self.batchnorm2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.batchnorm2(self.conv2(self.activation(self.batchnorm1(self.conv1(x)))))
        return self.activation(x + out)


class MLPBlock(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size)
        self.activation = torch.nn.SiLU()
        self.batchnorm = torch.nn.BatchNorm1d(size)
        self.dropout = torch.nn.Dropout(dropout_percent)

    def forward(self, x):
        x = x + self.dropout(self.activation(self.batchnorm(self.linear(x))))
        return x


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.piece_embed = torch.nn.Embedding(13, embed_size)
        self.res_blocks = torch.nn.Sequential(*[ResNetBlock(embed_size) for _ in range(n_res_blocks)])
        self.avg_pool = torch.nn.AvgPool2d(8)
        self.mlp_blocks = torch.nn.Sequential(*[MLPBlock(embed_size + 1) for _ in range(n_mlp_blocks)])
        self.final_linear = torch.nn.Linear(embed_size + 1, 3)

    def forward(self, x):
        global embed_size
        # B = batch size, M = embed size
        B = x.shape[0]
        turn = x[:, 0:1]  # (B, 1)
        # embedding vector for each piece type
        board = self.piece_embed(x[:, 1:]).view((B, embed_size, 8, 8))
        # apply a resnet and average pooling
        board = self.avg_pool(self.res_blocks(board)).view((B, embed_size))  # (B, M)
        # concatenate back the turn data
        x = torch.concat((turn, board), axis=1)  # (B, M + 1)
        # apply a MLP and then a final linear layer to get logits
        return self.final_linear(self.mlp_blocks(x))
