import transform


def test_get_move_to_int():
    move_to_int = transform.get_move_to_int()
    assert len(move_to_int) == 1968
    assert 'a1a5' in move_to_int
    assert 'b3d3' in move_to_int
    assert 'c1e3' in move_to_int
    assert 'g1f3' in move_to_int
    assert 'g7h8q' in move_to_int
    for i, m in enumerate(move_to_int):
        assert i == move_to_int[m]


def test_board_to_array():
    # starting position
    board = transform.chess.Board()
    n = transform.board_to_array(board)

    assert n.shape == (64,)
    assert type(n) == transform.numpy.ndarray
    assert n.dtype == transform.numpy.int64
    for x in n:
        assert 0 <= x <= 12
    assert n[0] == 10, "No black rook at a8"
    assert n[n.shape[0] - 2] == 2, "No white bishop at g1"


def test_get_time():
    assert transform.get_time('TimeControl "180"') == 180
    assert transform.get_time('TimeControl "240"') == 240
    assert transform.get_time('TimeControl "180+1"') == 180


def test_should_skip_game():
    assert transform.should_skip_game('TimeControl "60"')
    assert transform.should_skip_game('TimeControl "180". Game ended on time.')
    assert not transform.should_skip_game('TimeControl "180". Game ended by resignation.')
    assert transform.should_skip_game('TimeControl "180" [Event "Live Chess - Chess960"]')
    assert transform.should_skip_game('TimeControl "180" - game abandoned"]')
    assert transform.should_skip_game('TimeControl "180" [Event "Live Chess - Odds Chess"]')


def test_get_moves():
    assert transform.get_moves('1K b - -"] 1. e4 1... e5 2. Nf3 2... Nc6') == ['e4', 'e5', 'Nf3', 'Nc6']


def test_get_states():
    s = transform.get_states(['e3', 'e5'])
    assert s[0].dtype == transform.numpy.int64
    assert s[0].shape[0] == 66, "State is wrong length."

    assert s[0][1] == 0, "The first game state doesn't have white to move."
    assert s[0][s[0].shape[0] - 20] == 0, "No empty space on e3 at start"

    assert s[1][1] == 1, "The second game state doesn't have black to move."
    assert s[1][s[1].shape[0] - 20] == 1, "No white pawn on e3 after 1. e3"
