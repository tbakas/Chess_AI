from create_model import ChessModel
import pandas
import torch
import lightning
import numpy
import logging
import os


logger = logging.getLogger(__name__)
max_epochs = 1
batch_size = 64
patience = 10
nrows = 2000
base_path = ''


class ChessLightning(lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        logits = self.model(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
        optim_dict = {'optimizer': optimizer,
                      'lr_schedule': {'scheduler': lr_scheduler, 'step': 'epoch', 'frequency': 1}}
        return optim_dict

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        logits = self.model(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        self.log('val_loss', loss)

        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == y_batch).sum() / y_batch.shape[0]
        self.log('val_acc', acc)

        return loss, acc

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        logits = self.model(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        self.log('test_loss', loss)

        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == y_batch).sum() / y_batch.shape[0]
        self.log('test_acc', acc)


def train_val_test_split(u: torch.Tensor, v: torch.Tensor, train_perc: float, val_perc: float) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # assumes data is already shuffled
    assert u.shape[0] == v.shape[0], 'Input tensors must have identical first dimensions'
    assert 0 < train_perc < 1 and 0 < val_perc <= 1 - train_perc, 'Invalid percentages'
    n = round(train_perc * u.shape[0])
    m = round(val_perc * u.shape[0])
    return u[:n], u[n: n + m], u[n + m:], v[:n], v[n: n + m], v[n + m:]


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
    data = pandas.read_csv(base_path + "transformed_data.csv", skiprows=1, nrows=nrows).to_numpy()

    idx = numpy.arange(data.shape[0])
    rng = numpy.random.default_rng(seed=27)
    rng.shuffle(idx)
    data = data[idx]

    y = torch.tensor(data[:, 0])
    x = torch.tensor(data[:, 1:])
    logger.info(f'x.shape is {x.shape}, and y.shape is {y.shape}')
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, train_perc=0.8, val_perc=0.1)

    del data, rng, idx, x, y

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    chess_lightning = ChessLightning(chess_model)
    early_stopping = lightning.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',
                                                               verbose=False)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                                      filename="{epoch}-{val_loss:.3f}")

    trainer = lightning.Trainer(max_epochs=max_epochs, callbacks=[early_stopping, checkpoint_callback])

    logger.info('Beginning the training loop...')
    trainer.fit(chess_lightning, train_loader, val_loader)
    logger.info('Training completed.')

    del train_loader, train_dataset, val_dataset, val_loader

    # test with the best model version
    lightning_logs = os.listdir('lightning_logs')
    version = lightning_logs[-1]
    checkpoint = os.listdir('lightning_logs/' + version + '/checkpoints')[0]
    checkpoint_path = 'lightning_logs/' + version + '/checkpoints/' + checkpoint
    logger.info('Loading checkpoint titled ' + checkpoint + ' in ' + version)
    chess_lightning = ChessLightning.load_from_checkpoint(checkpoint_path, model=ChessModel())

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info('Beginning testing.')
    trainer.test(chess_lightning, test_loader)
