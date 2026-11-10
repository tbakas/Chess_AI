from create_model import ChessModel
import torch
import lightning
import numpy
import logging
import os


logger = logging.getLogger(__name__)
max_epochs = 10
max_rows = 5000
batch_size = 512
base_path = ''


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


def train_val_test_split(size: int, train_perc: float, val_perc: float) -> tuple[int, int]:
    # assumes data is already shuffled
    assert 0 < train_perc < 1 and 0 < val_perc <= 1 - train_perc, 'Invalid percentages'
    n_ = round(train_perc * size)
    m_ = n_ + round(val_perc * size)
    return n_, m_


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
    data = numpy.load(base_path + "transformed_data.npz")
    x = torch.tensor(data['position'][:max_rows])
    t = torch.tensor(data['turn'][:max_rows])
    y = torch.tensor(data['next_move'][:max_rows])
    mask = torch.tensor(data['mask'][:max_rows])

    del data

    # shuffle data
    idx = numpy.arange(x.shape[0])
    rng = numpy.random.default_rng(seed=24)
    rng.shuffle(idx)
    x, t, y, mask = x[idx], t[idx], y[idx], mask[idx]

    del rng, idx

    n, m = train_val_test_split(x.shape[0], 0.8, 0.1)
    x_train, x_val, x_test = x[:n], x[n:m], x[m:]
    t_train, t_val, t_test = t[:n], t[n:m], t[m:]
    y_train, y_val, y_test = y[:n], y[n:m], y[m:]
    mask_train, mask_val, mask_test = mask[:n], mask[n:m], mask[m:]

    logger.info(f'x.shape is {x.shape}, t.shape is {t.shape}, y.shape is {y.shape} and mask.shape is {mask.shape}')

    del x, y, t, mask

    train_dataset = torch.utils.data.TensorDataset(x_train, t_train, y_train, mask_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(x_val, t_val, y_val, mask_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    chess_lightning = ChessLightning(chess_model)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                                      filename="{epoch}-{val_loss:.3f}")

    trainer = lightning.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback])

    logger.info('Beginning the training loop...')
    trainer.fit(chess_lightning, train_loader, val_loader)
    logger.info('Training completed.')

    del train_loader, train_dataset, val_dataset, val_loader, x_train, x_val, t_train, t_val, y_train, y_val
    del mask_train, mask_val

    # test with the best model version
    lightning_logs = os.listdir('lightning_logs')
    version = lightning_logs[-1]
    checkpoint = os.listdir('lightning_logs/' + version + '/checkpoints')[0]
    checkpoint_path = 'lightning_logs/' + version + '/checkpoints/' + checkpoint
    logger.info('Loading checkpoint titled ' + checkpoint + ' in ' + version)
    chess_lightning = ChessLightning.load_from_checkpoint(checkpoint_path, model=ChessModel())

    test_dataset = torch.utils.data.TensorDataset(x_test, t_test, y_test, mask_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info('Beginning testing.')
    trainer.test(chess_lightning, test_loader)
