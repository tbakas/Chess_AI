import chess
import numpy
from pandas import DataFrame
from re import search
from os import listdir
from os.path import isfile
import logging


logger = logging.getLogger(__name__)
piece_to_int = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
# Upper case are white pieces, lower case are black pieces


def board_to_array(b: chess.Board) -> numpy.ndarray:
    n = numpy.zeros(64, dtype=int)
    i = 0
    for c in b.__str__():
        if c == ' ' or c == '\n':
            continue
        n[i] = piece_to_int[c]
        i += 1
    return n


def get_result(text: str) -> int:
    pattern = '1-0|0-1|1/2-1/2'
    m = search(pattern, text).group()
    if m == '1-0':
        return 0
    elif m == '0-1':
        return 2
    return 1


def get_time(text: str) -> int:
    m = search('TimeControl "\\d+', text).group()
    return int(m.split()[-1].strip('"'))


def should_skip_game(game_text: str) -> bool:
    time = get_time(game_text)
    if time < 180:
        return True
    if search('Chess960', game_text):
        return True
    if search('on time', game_text):
        return True
    if search('game abandoned', game_text):
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


def get_states(result: int, moves: list[str]) -> list[numpy.ndarray]:
    # doesn't include starting position
    board = chess.Board()
    s = []
    x = numpy.array([result, 1])
    for move in moves:
        board.push_san(move)
        s.append(numpy.concat((x, board_to_array(board)), dtype=numpy.int64))
        x[1] ^= 1
    return s


def game_states_generator(file_paths: list[str]) -> list[numpy.ndarray]:
    for file_name in file_paths:
        logger.info(f'Opening file {file_name}.')
        try:
            with open(file_name, 'r') as file:
                games = file.read().split('\n\n\n')
                for c, game in enumerate(games):
                    try:
                        if should_skip_game(game):
                            continue
                        result = get_result(game)
                        move_list = get_moves(game)
                    except AttributeError:
                        error_message = f'Game {c} in file {file_name} had a regex search error.'
                        logger.error(error_message + '\n' + game)
                        continue

                    try:
                        s = get_states(result, move_list)
                    except (chess.IllegalMoveError, chess.InvalidMoveError):
                        error_message = f'Game {c} in file {file_name} has an illegal move in move list.'
                        logger.error(error_message + '\n' + game)
                        continue

                    yield s
        except FileNotFoundError:
            logger.exception(f'The file {file_name} does not exist.')


if __name__ == '__main__':
    logging.basicConfig(filename='transform_log.log',
                        format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

    data = DataFrame(columns=['result', 'turn'] + [str(i) for i in range(64)])
    # result: 0 for white win, 1 for draw and 2 for black win
    # turn: 0 for white's turn, 1 for black's turn
    # remaining columns are for recording where pieces are located on the board

    game_count = 0
    data_entries = 0
    csv_path = 'data/transformed_data.csv'
    file_exists = isfile(csv_path)

    logger.info('Starting the chess data transformation pipeline...')

    paths = ['data/pgn_files/' + x for x in listdir('data/pgn_files/')]
    states_list = game_states_generator(paths)
    for states in states_list:
        print(f'\rgame count: {game_count}', end='')
        for state in states:
            # iterate through all board states in a certain game and add them to our dataframe
            data.loc[data.shape[0]] = state
            data_entries += 1

            if data.shape[0] >= 10000:
                # save current data and empty the dataframe to prevent memory crash
                data.to_csv(csv_path, mode='a', index=False, header=not file_exists)
                file_exists = True
                data = DataFrame(columns=['result', 'turn'] + [str(i) for i in range(64)])

        game_count += 1
        if game_count % 100 == 0:
            logging.info(f'Successfully parsed {game_count} games with {data_entries} total entries.')

    data.to_csv(csv_path, mode='a', index=False, header=False)
    logger.info(f'Finished data transformation. Saved {game_count} games with {data_entries} data points.')
