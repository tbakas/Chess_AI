import pytest
import torch
import create_model
import train_model


def test_training_epoch():
    model = create_model.ChessModel()
    x = torch.randint(high=13, size=(5, 65))
    y = torch.randint(high=3, size=(5,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters())
    avg_loss = train_model.training_epoch(model, optimizer, loader)

    assert type(avg_loss) == float
    assert not torch.isnan(torch.tensor(avg_loss))

    assert model.training

    for name, param in model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has null gradient.'
        assert not torch.isnan(param.grad).any(), f'Parameter {name} has a nan value in gradient.'
        assert not torch.equal(param.grad, torch.zeros(param.grad.shape))


def test_validation_epoch():
    model = create_model.ChessModel()
    x = torch.randint(high=13, size=(5, 65))
    y = torch.randint(high=3, size=(5,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    avg_loss, acc, predictions = train_model.validation_epoch(model, loader, prediction_size=y.shape[0])

    assert type(avg_loss) == float
    assert not torch.isnan(torch.tensor(avg_loss))

    assert type(acc) == float
    assert not torch.isnan(torch.tensor(acc))
    assert 0 <= acc <= 100

    vals = {0, 1, 2}
    assert y.shape == predictions.shape
    for i in range(predictions.shape[0]):
        assert predictions[i].item() in vals

    assert not model.training


def test_train_val_test_split():
    x = torch.randint(high=13, size=(10, 65))
    y = torch.randint(high=3, size=(10,))

    x_train, x_val, x_test, y_train, y_val, y_test = train_model.train_val_test_split(x, y, 0.8, 0.1)
    assert x_train.shape == (8, 65)
    assert x_val.shape == (1, 65)
    assert x_test.shape == (1, 65)
    assert y_train.shape == (8,)
    assert y_val.shape == (1,)
    assert y_test.shape == (1,)
