import torch
from create_model import ChessModel


def test_output():
    model = ChessModel()
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randint(high=13, size=(2, 65))
        out = model(dummy_input)

    assert out.shape == (2, 1968), f'Incorrect shape of {out.shape}, expected (2, 1968).'
    assert out.device == dummy_input.device
    assert out.dtype == torch.float32


def test_gradient():
    model = ChessModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad()
    dummy_input = torch.randint(high=13, size=(2, 65))
    out = model(dummy_input).sum()
    out.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has null gradient.'
        assert not torch.isnan(param.grad).any(), f'Parameter {name} has a nan value in gradient.'
        assert not torch.equal(param.grad, torch.zeros(param.grad.shape))


def test_serialization():
    model = ChessModel()
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randint(high=13, size=(2, 65))
        out1 = model(dummy_input)

    torch.save(model.state_dict(), 'dummy_model.pt')

    model = ChessModel()
    model.load_state_dict(torch.load('dummy_model.pt'))
    model.eval()
    with torch.no_grad():
        out2 = model(dummy_input)

    torch.testing.assert_close(out1, out2)
