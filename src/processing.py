import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

import encoder

C = 13

PIECE_TO_PLANE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

base_dir = Path(__file__).parent.parent  # folder where this script lives
output_path = base_dir / "datasets" / "processed" / "chess_dataset.npz"
# Make sure directories exist
output_path.parent.mkdir(parents=True, exist_ok=True)

VALUE_LABEL = {"white": 1, "draw": 0, "black": -1}

INPUT_SHAPE = (C, 8, 8)


def vectorise(fen):
    board, to_move = fen.split()[:2]

    tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)

    rows = board.split("/")

    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                plane = PIECE_TO_PLANE[char]
                tensor[plane, i, j] = 1.0
                j += 1

    if to_move == "w":
        tensor[12, :, :] = 1.0

    return tensor


def parse_data_set(df):
    t_table = {}  ##(fen_str -> nparray)
    dataset_len = 0
    states = []  # Tensors
    policy = []  # prediction lables
    values = []  # result labels

    print("processing boards")
    row_count = 0
    for row in df.iter_rows(named=True):
        row_count += 1
        # print(f"processing row: {row_count}")
        moves = row["moves"].split(" ")
        n_moves = len(moves)
        dataset_len += n_moves
        values += [VALUE_LABEL[row["winner"]] for _ in range(n_moves)]

        # vectorise state walk
        board = chess.Board()
        for move in moves:
            fen = board.fen()
            san_mv = board.parse_san(move)
            index = encoder.encode_az_4672(san_mv)
            policy.append(index)

            if fen in t_table:
                states.append(t_table[fen])
            else:
                tensor = vectorise(fen)
                t_table[fen] = tensor
                states.append(tensor)

            board.push(san_mv)

    states = np.stack(states)
    policy = np.array(policy, dtype=np.int64)
    values = np.array(values, dtype=np.float32)

    print("writing to .npz")
    np.savez_compressed(
        output_path,
        states=states,
        policy=policy,
        values=values,
    )


if __name__ == "__main__":
    starting_time = time.time()
    FILE_PATH = "../datasets/raw/games.csv"
    df = pl.read_csv(FILE_PATH, columns=["moves", "winner"])
    df = df.filter(df["winner"].is_in(["white", "black", "draw"]))

    parse_data_set(df)
    print(f"data processing took:\n{starting_time - time.time()} seconds")
