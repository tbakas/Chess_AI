import torch

kernel_size = 3
dropout_percent = 0.5
embed_size = 128
n_res_blocks = 4
n_mlp_blocks = 10
mlp_size = 32
output_size = 1968


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


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.piece_embed = torch.nn.Embedding(13, embed_size)
        self.res_blocks = torch.nn.Sequential(*[ResNetBlock(embed_size) for _ in range(n_res_blocks)])
        self.conv1x1 = torch.nn.Conv2d(embed_size, 8, kernel_size=1)
        self.final_linear = torch.nn.Linear(512 + 1, output_size)
        self.batchnorm = torch.nn.BatchNorm2d(8)
        self.activation = torch.nn.SiLU()

    def forward(self, x, turn):
        global embed_size
        # B = batch size, M = embed size
        B = x.shape[0]
        turn = turn.view((B, 1))
        # embedding vector for each piece type
        x = self.piece_embed(x).view((B, embed_size, 8, 8))
        # apply a resnet
        x = self.res_blocks(x)  # (B, M, 8, 8)
        # send through a 1x1 convolution with batchnorm/activation then flatten
        x = self.activation(self.batchnorm(self.conv1x1(x))).view((B, -1))  # (B, 512)
        # concat with turn information
        x = torch.concat((x, turn), axis=-1)  # (B, 513)
        # linear layer to logits
        return self.final_linear(x)


if __name__ == '__main__':
    model = ChessModel()
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')