from create_model import ChessModel
import torch
import lightning
import numpy
import logging


logger = logging.getLogger(__name__)
max_epochs = 1
batch_size = 128
base_path = ''
train_perc = 0.8
val_perc = 0.1


class ChessLightning(lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
        optim_dict = {'optimizer': optimizer,
                      'lr_schedule': {'scheduler': lr_scheduler, 'step': 'epoch', 'frequency': 1}}
        return optim_dict

    def calculate_loss(self, batch, get_acc=False):
        x_batch, t_batch, y_batch, m_batch = batch
        masked_logits = self.model(x_batch, t_batch).masked_fill(m_batch == 0, float('-inf'))
        loss = torch.nn.functional.cross_entropy(masked_logits, y_batch)

        if get_acc:
            _, predicted = torch.max(masked_logits.data, 1)
            acc = (predicted == y_batch).sum() / y_batch.shape[0]
        else:
            acc = None
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self.calculate_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.calculate_loss(batch, get_acc=True)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.calculate_loss(batch, get_acc=True)
        self.log('test_loss', loss)
        self.log('test_acc', acc)


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, x_path, t_path, y_path, mask_path):
        self.x = numpy.load(x_path, mmap_mode='r')
        self.t = numpy.load(t_path, mmap_mode='r')
        self.y = numpy.load(y_path, mmap_mode='r')
        self.packed_mask = numpy.load(mask_path, mmap_mode='r')

    def __len__(self):
        return len(self.t)

    def __getitem__(self, item):
        x_tensor = torch.tensor(self.x[item], dtype=torch.long)
        t_tensor = torch.tensor(self.t[item], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[item], dtype=torch.long)
        mask_tensor = torch.tensor(numpy.unpackbits(self.packed_mask[item], axis=0), dtype=torch.bool)
        return x_tensor, t_tensor, y_tensor, mask_tensor


if __name__ == '__main__':
    logging.basicConfig(filename='train_log.log',
                        format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'The device used is {device}')
    chess_model = ChessModel().to(device)

    logger.info('Opening transformed_data.csv')
    dataset = ChessDataset(base_path + 'x.npy', base_path + 't.npy', base_path + 'y.npy', base_path + 'mask.npy')

    indices = numpy.arange(len(dataset))
    n = round(train_perc * indices.shape[0])
    m = n + round(val_perc * indices.shape[0])
    rng = numpy.random.default_rng()
    rng.shuffle(indices)

    train_dataset = torch.utils.data.Subset(dataset, indices[:n])
    val_dataset = torch.utils.data.Subset(dataset, indices[n:m])
    test_dataset = torch.utils.data.Subset(dataset, indices[m:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    chess_lightning = ChessLightning(chess_model)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                                      filename="{epoch}-{val_loss:.3f}",
                                                                      dirpath=base_path)

    trainer = lightning.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback])

    logger.info('Beginning the training loop...')
    trainer.fit(chess_lightning, train_loader, val_loader)
    logger.info('Training completed.')

    del train_loader, train_dataset, val_dataset, val_loader

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info('Beginning testing.')
    trainer.test(chess_lightning, test_loader, ckpt_path='best')
