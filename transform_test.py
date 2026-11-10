import chess

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


def test_board_to_list():
    # starting position
    board = transform.chess.Board()
    n = transform.board_to_list(board)

    assert len(n) == 64
    assert all(isinstance(x, int) for x in n)
    for x in n:
        assert 0 <= x <= 12
    assert n[0] == 10, "No black rook at a8"
    assert n[len(n) - 2] == 2, "No white bishop at g1"


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


def test_transform_move_list():
    pos, turn, n_move, mask = transform.transform_move_list(['e3', 'e5'])
    assert all(isinstance(x, int) for x in pos[0])
    assert len(pos[0]) == 64 and len(pos[1]) == 64
    assert len(pos) == 2

    for t in turn:
        assert t == 0 or t == 1
    for b in mask[0]:
        assert b == 0 or b == 1
    for m in n_move:
        assert 0 <= m <= 1967 and type(m) == int

    board = chess.Board()

    assert turn[0] == 0, "The first game state doesn't have white to move."
    assert pos[0][64 - 20] == 0, "No empty space on e3 at start"
    assert n_move[0] == transform.move_to_int[board.parse_san('e3').uci()]
    for move in board.legal_moves:
        n = transform.move_to_int[move.uci()]
        assert mask[0][n] == 1
    assert sum(mask[0]) == board.legal_moves.count()

    board.push_san('e3')

    assert turn[1] == 1, "The second game state doesn't have black to move."
    assert pos[1][64 - 20] == 1, "No white pawn on e3 after 1. e3"
    assert n_move[1] == transform.move_to_int[board.parse_san('e5').uci()]
    print(transform.move_to_int)
    for move in board.legal_moves:
        n = transform.move_to_int[move.uci()]
        assert mask[1][n] == 1
    assert sum(mask[1]) == board.legal_moves.count()
