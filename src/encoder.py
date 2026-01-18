import chess

# Promotion options: None=0, Queen=1, Rook=2, Bishop=3, Knight=4
PROMO_MAP = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}

SLIDING_DIRS = [
    (-1, 0),  # N
    (1, 0),  # S
    (0, 1),  # E
    (0, -1),  # W
    (-1, 1),  # NE
    (-1, -1),  # NW
    (1, 1),  # SE
    (1, -1),  # SW
]

KNIGHT_DIRS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

UNDERPROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def encode_az_4672(move):
    from_sq = move.from_square
    to_sq = move.to_square

    assert 0 <= move.from_square < 64
    assert 0 <= move.to_square < 64

    fr, fc = divmod(from_sq, 8)
    tr, tc = divmod(to_sq, 8)

    dr = tr - fr
    dc = tc - fc

    if move.promotion in UNDERPROMOS:
        promo_idx = UNDERPROMOS.index(move.promotion)
        # forward, forward-left, forward-right
        if dc == 0:
            dir_idx = 0
        elif dc == -1:
            dir_idx = 1
        elif dc == 1:
            dir_idx = 2
        else:
            raise ValueError("Invalid underpromotion direction")

        move_type = 64 + dir_idx * 3 + promo_idx
        assert 0 <= move_type < 73

        ans = from_sq * 73 + move_type
        return ans

    if (dr, dc) in KNIGHT_DIRS:
        move_type = 56 + KNIGHT_DIRS.index((dr, dc))
        assert 0 <= move_type < 73
        return from_sq * 73 + move_type

    for dir_idx, (sdr, sdc) in enumerate(SLIDING_DIRS):
        for dist in range(1, 8):
            if dr == sdr * dist and dc == sdc * dist:
                move_type = dir_idx * 7 + (dist - 1)
                assert 0 <= move_type < 73
                return from_sq * 73 + move_type

    raise ValueError(f"Unencodable move: {move}")


def decode_az_4672(index):
    from_sq = index // 73
    move_type = index % 73
    fr, fc = divmod(from_sq, 8)

    to_sq = None
    promotion = None

    if 64 <= move_type < 73:  # underpromotion
        up_idx = move_type - 64
        dir_idx, promo_idx = divmod(up_idx, 3)
        tc = fc
        if dir_idx == 1:
            tc = fc - 1
        elif dir_idx == 2:
            tc = fc + 1
        tr = fr + 1
        promotion = UNDERPROMOS[promo_idx]
        to_sq = tr * 8 + tc

    elif 56 <= move_type < 64:  # knight
        k_idx = move_type - 56
        dr, dc = KNIGHT_DIRS[k_idx]
        tr, tc = fr + dr, fc + dc
        to_sq = tr * 8 + tc

    else:  # sliding
        for dir_idx, (sdr, sdc) in enumerate(SLIDING_DIRS):
            for dist in range(1, 8):
                if dir_idx * 7 + (dist - 1) == move_type:
                    tr, tc = fr + sdr * dist, fc + sdc * dist
                    to_sq = tr * 8 + tc
                    break

    # --- centralized clipping ---
    if to_sq is None or not (0 <= from_sq < 64) or not (0 <= to_sq < 64):
        return None

    return chess.Move(from_sq, to_sq, promotion=promotion)
