import chess
import numpy
from re import search
from os import listdir
import logging


logger = logging.getLogger(__name__)
game_max = None
piece_to_int = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
# Upper case are white pieces, lower case are black pieces


def get_move_to_int() -> dict[str, int]:
    moves = []

    for x in range(8):
        for y in range(8):
            # Queen type moves
            board = chess.Board("8/8/8/8/8/8/8/8")
            board.set_piece_at(chess.square(x, y), chess.Piece(chess.QUEEN, chess.WHITE))
            moves.extend(board.generate_legal_moves())

            # Knight moves
            board = chess.Board("8/8/8/8/8/8/8/8")
            board.set_piece_at(chess.square(x, y), chess.Piece(chess.KNIGHT, chess.WHITE))
            moves.extend(board.generate_legal_moves())

    # Pawn promotions
    for x in range(8):
        for y in [1, 6]:
            board = chess.Board("8/8/8/8/8/8/8/8")

            if y == 6:
                # white to promote
                dy = 1
                color = chess.WHITE
                other = chess.BLACK
            else:
                # black to promote
                dy = -1
                other = chess.WHITE
                color = chess.BLACK
            board.set_piece_at(chess.square(x, y), chess.Piece(chess.PAWN, color))
            board.turn = color

            if x <= 6:
                board.set_piece_at(chess.square(x + 1, y + dy), chess.Piece(chess.ROOK, other))
            if x >= 1:
                board.set_piece_at(chess.square(x - 1, y + dy), chess.Piece(chess.ROOK, other))
            moves.extend(board.generate_legal_moves())

    move_to_int_ = dict()
    for i, m in enumerate(moves):
        move_to_int_[m.uci()] = i
    return move_to_int_


move_to_int = get_move_to_int()


def board_to_list(b: chess.Board) -> list[int]:
    n = [0] * 64
    i = 0
    for c in b.__str__():
        if c == ' ' or c == '\n':
            continue
        n[i] = piece_to_int[c]
        i += 1
    return n


def get_time(text: str) -> int:
    m = search('TimeControl "\\d+', text).group()
    return int(m.split()[-1].strip('"'))


def should_skip_game(game_text: str) -> bool:
    time = get_time(game_text)
    if time < 180:
        return True
    pattern = 'Chess960|on time|game abandoned|Odds Chess'
    if search(pattern, game_text):
        return True
    return False


def get_moves(game_text: str) -> list[str]:
    match = search(r'1\. ', game_text)
    game_text = game_text[match.end():]

    move_list = []
    for move in game_text.split():
        if move[0].isalpha():
            move_list.append(move)
    return move_list


def transform_move_list(moves: list[str]) -> tuple[list[list[int]], list[int], list[int], list[list[int]]]:
    global move_to_int
    board = chess.Board()
    position_ = []
    turn_ = []
    next_move_ = []
    mask_ = []
    t = 0
    for move in moves:
        move = board.parse_san(move)
        position_.append(board_to_list(board))
        turn_.append(t)
        next_move_.append(move_to_int[move.uci()])

        dummy = [0] * 1968
        for legal_move in board.legal_moves:
            dummy[move_to_int[legal_move.uci()]] = 1
        mask_.append(dummy)

        board.push(move)
        t ^= 1
    return position_, turn_, next_move_, mask_


def game_data_generator(file_paths: list[str]) -> tuple[list[list[int]], list[int], list[int], list[list[int]]]:
    for file_name in file_paths:
        logger.info(f'Opening file {file_name}.')
        try:
            with open(file_name, 'r') as file:
                games = file.read().split('\n\n\n')
                for c, game in enumerate(games):
                    try:
                        if should_skip_game(game):
                            continue
                        move_list = get_moves(game)
                    except AttributeError:
                        error_message = f'Game {c} in file {file_name} had a regex search error.'
                        logger.error(error_message + '\n' + game + '\n')
                        continue

                    try:
                        out = transform_move_list(move_list)
                    except (chess.IllegalMoveError, chess.InvalidMoveError):
                        error_message = f'Game {c} in file {file_name} has an illegal move in move list.'
                        logger.error(error_message + '\n' + game + '\n')
                        continue

                    yield out
        except FileNotFoundError:
            logger.exception(f'The file {file_name} does not exist.')


if __name__ == '__main__':
    logging.basicConfig(filename='transform_log.log',
                        format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

    position = []
    # position: 1x64 array representing the board with an integer in range (0, 13) for each piece type
    turn = []
    # turn: 0 for white's turn, 1 for black's turn
    next_move = []
    # next_move: an integer in range(0, 1968) for the move played in current position
    mask = []
    # mask: 1x1968 boolean array for recording what moves where legal in position

    game_count = 0
    data_entries = 0
    logger.info('Starting the chess data transformation pipeline...')

    paths = ['data/' + x for x in listdir('data/')]
    game_data_list = game_data_generator(paths)
    for game_data in game_data_list:
        print(f'\rgame count: {game_count}', end='')
        position.extend(game_data[0])
        turn.extend(game_data[1])
        next_move.extend(game_data[2])
        mask.extend(game_data[3])

        game_count += 1

        if game_count % 500 == 0:
            logging.info(f'Successfully parsed {game_count} games with {data_entries} total entries.')
        if game_max and game_count >= game_max:
            break

    numpy.savez_compressed('transformed_data.npz', position=position, turn=turn, next_move=next_move, mask=mask)
    logger.info(f'Finished data transformation. Saved {game_count} games with {data_entries} data points.')
