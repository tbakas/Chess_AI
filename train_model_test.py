import torch
import create_model
import train_model


def test_calculate_loss():
    model = create_model.ChessModel()
    model_lightning = train_model.ChessLightning(model)
    x = torch.randint(high=13, size=(5, 64))
    t = torch.randint(high=2, size=(5,))
    y = torch.randint(high=1968, size=(5,))
    mask = torch.randint(high=2, size=(5, 1968))

    avg_loss, acc = model_lightning.calculate_loss((x, t, y, mask), get_acc=True)

    assert type(avg_loss) == torch.Tensor
    assert avg_loss.dtype == torch.float
    assert not torch.isnan(avg_loss)
    assert 0 <= avg_loss.item() < float('inf')

    assert type(acc) == torch.Tensor
    assert acc.dtype == torch.float
    assert not torch.isnan(acc)
    assert 0 <= acc.item() <= 100


def test_train_val_test_split():
    n, m = train_model.train_val_test_split(10, 0.8, 0.1)
    assert n == 8
    assert m == 9
