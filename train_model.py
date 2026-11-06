from create_model import ChessModel
import pandas
import torch
import lightning
import numpy
import logging
from sklearn.metrics import confusion_matrix
import os


logger = logging.getLogger(__name__)
max_epochs = 3
batch_size = 64
# patience parameter for early stopping
patience = 3
# if set to false, this makes sure not all cpu cores are used
on_collab = False


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


def get_test_results(model: ChessModel, loader: torch.utils.data.DataLoader, prediction_size) \
        -> tuple[float, float, torch.Tensor]:
    predictions = torch.Tensor([0] * prediction_size)
    eval_loss = 0.0
    correct = k = 0
    model.eval()

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            eval_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == yb).sum().item()

            if prediction_size:
                predictions[k: k + predicted.shape[0]] = predicted
                k += predicted.shape[0]

    # make sure all the prediction tensor's elements were modified
    logger.info(f'Test predictions had {k} / {prediction_size} updated values.')
    return eval_loss / len(loader), 100.0 * correct / prediction_size, predictions


def train_val_test_split(u: torch.Tensor, v: torch.Tensor, train_perc: float, val_perc: float) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    logger.info('Opening data/transformed_data.csv')
    data = pandas.read_csv("data/transformed_data.csv", header=None, nrows=2000).to_numpy()

    idx = numpy.arange(data.shape[0])
    rng = numpy.random.default_rng(seed=27)
    rng.shuffle(idx)
    data = data[idx]

    y = torch.tensor(data[:, 0], device=device)
    x = torch.tensor(data[:, 1:], device=device)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, train_perc=0.8, val_perc=0.1)

    del data, rng, idx, x, y

    if not on_collab:
        num_workers = max(1, os.cpu_count() - 1)
    else:
        num_workers = max(1, os.cpu_count())

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, persistent_workers=True)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, persistent_workers=True)

    chess_model = ChessModel().to(device)
    chess_lightning = ChessLightning(chess_model)
    early_stopping = lightning.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',
                                                               verbose=False)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                                      filename="{epoch}-{val_loss:.3f}")

    trainer = lightning.Trainer(max_epochs=max_epochs, callbacks=[early_stopping, checkpoint_callback])

    logger.info('Beginning the training loop...')
    trainer.fit(chess_lightning, train_loader, val_loader)
    logger.info('Training completed.')

    del trainer, train_loader, train_dataset, val_dataset, val_loader

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, persistent_workers=True)

    logger.info('Beginning testing.')
    test_loss, test_acc, y_hat = get_test_results(chess_model, test_loader, prediction_size=y_test.shape[0])
    del test_loader, test_dataset

    labels = ['1-0', '1/2-1/2', '0-1']
    confusion = pandas.DataFrame(confusion_matrix(y_pred=y_hat, y_true=y_test), index=labels, columns=labels)
    confusion = '\nConfusion Table. Rows are true values. Columns are predictions.\n' + confusion.to_string()
    logger.info(f'Test loss of {test_loss: .3f}. Test accuracy of {test_acc: .2f}%' + confusion)
