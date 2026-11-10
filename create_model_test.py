import torch
from create_model import ChessModel


def test_output():
    model = ChessModel()
    model.eval()
    with torch.no_grad():
        dummy_x = torch.randint(high=13, size=(2, 64))
        dummy_t = torch.randint(high=2, size=(2, 1))
        out = model(dummy_x, dummy_t)

    assert out.shape == (2, 1968), f'Incorrect shape of {out.shape}, expected (2, 1968).'
    assert out.device == dummy_x.device
    assert out.dtype == torch.float32


def test_gradient():
    model = ChessModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad()
    dummy_x = torch.randint(high=13, size=(2, 64))
    dummy_t = torch.randint(high=2, size=(2, 1))
    out = model(dummy_x, dummy_t).sum()
    out.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has null gradient.'
        assert not torch.isnan(param.grad).any(), f'Parameter {name} has a nan value in gradient.'
        assert not torch.equal(param.grad, torch.zeros(param.grad.shape))


def test_serialization():
    model = ChessModel()
    model.eval()
    with torch.no_grad():
        dummy_x = torch.randint(high=13, size=(2, 64))
        dummy_t = torch.randint(high=2, size=(2, 1))
        out1 = model(dummy_x, dummy_t)

    torch.save(model.state_dict(), 'dummy_model.pt')

    model = ChessModel()
    model.load_state_dict(torch.load('dummy_model.pt'))
    model.eval()
    with torch.no_grad():
        out2 = model(dummy_x, dummy_t)

    torch.testing.assert_close(out1, out2)
