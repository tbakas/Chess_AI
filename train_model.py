from create_model import ChessModel
import pandas
import torch
import numpy
from matplotlib import pyplot
import logging
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


def training_epoch(model: ChessModel, optimizer: torch.optim.Optimizer, loader: torch.utils.data.DataLoader) -> float:
    epoch_loss = 0.0
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    return epoch_loss / len(loader)


def validation_epoch(model: ChessModel, loader: torch.utils.data.DataLoader, prediction_size=0) \
        -> tuple[float, float, torch.Tensor]:
    predictions = torch.Tensor([0] * prediction_size)
    eval_loss = 0.0
    correct = k = 0
    count = len(loader) * loader.batch_size
    model.eval()

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            eval_loss += loss.item()

            probabilities = torch.softmax(logits, dim=-1)
            _, predicted = torch.max(probabilities.data, 1)
            correct += (predicted == yb).sum().item()

            if prediction_size:
                predictions[k: k + predicted.shape[0]] = predicted
                k += predicted.shape[0]

    if prediction_size:
        logger.info(f'Test predictions had {k} / {prediction_size} updated values.')

    return eval_loss / len(loader), 100.0 * correct / count, predictions


def train_val_test_split(u: torch.Tensor, v: torch.Tensor, train_perc: float, val_perc: float) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert u.shape[0] == v.shape[0], 'Input tensors must have identical first dimensions'
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
    batch_size = 64
    epochs = 40

    logger.info('Opening data/transformed_data.csv...')
    data = pandas.read_csv("data/transformed_data.csv", header=None).to_numpy()

    idx = numpy.arange(data.shape[0])
    rng = numpy.random.default_rng(seed=27)
    rng.shuffle(idx)
    data = data[idx]

    y = torch.tensor(data[:, 0], device=device)
    x = torch.tensor(data[:, 1:], device=device)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, train_perc=0.8, val_perc=0.1)

    del data, rng, idx, x, y

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    chess_optimizer = torch.optim.AdamW(chess_model.parameters(), lr=1e-4)
    train_loss = []
    val_loss = []
    val_acc = []
    min_loss = float('inf')

    logger.info('Beginning the training loop...')
    for i in range(epochs):
        print(f'\rCompleted {i} out of {epochs} epochs.', end='')
        chess_loss = training_epoch(chess_model, chess_optimizer, train_loader)
        train_loss.append(chess_loss)

        chess_loss, chess_acc, _ = validation_epoch(chess_model, val_loader)
        val_loss.append(chess_loss)
        val_acc.append(chess_acc)

        message1 = f'Completed epoch {i + 1}. Training loss of {train_loss[-1]: .3f}, val loss of {chess_loss: .3f}, '
        message2 = f'and val accuracy of {chess_acc: .3f}%'
        logger.info(message1 + message2)

        if val_loss[-1] < min_loss:
            min_loss = val_loss[-1]
            torch.save(chess_model.state_dict(), "model_weights.pth")
            torch.save(chess_optimizer.state_dict(), "optimizer.pth")
            logger.info(f'Obtained a minium val loss of {min_loss: .3f} on epoch {i + 1}. Saving current model.')

    del chess_optimizer, train_loader, train_dataset, val_dataset, val_loader

    logger.info(f'Training complete with a minimum val loss of {min_loss: .3f}')

    dummy = range(1, epochs + 1)
    pyplot.plot(dummy, train_loss, 'o-')
    pyplot.plot(dummy, val_loss, 'o-')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['Train', 'Val'])
    pyplot.savefig('loss_plot.png')

    pyplot.clf()
    pyplot.plot(dummy, val_acc, 'o-')
    pyplot.xlabel('epoch')
    pyplot.ylabel('val acc')
    pyplot.savefig('acc_plot.png')

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc, y_hat = validation_epoch(chess_model, test_loader, prediction_size=y_test.shape[0])

    labels = ['1-0', '1/2-1/2', '0-1']
    confusion = pandas.DataFrame(confusion_matrix(y_pred=y_hat, y_true=y_test), index=labels, columns=labels)
    confusion = '\nConfusion Table. Rows are true values. Columns are predictions.\n' + confusion.to_string()
    logger.info(f'Test loss of {test_loss: .3f}. Test accuracy of {test_acc: .3f}%' + confusion)
