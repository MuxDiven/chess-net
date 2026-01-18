from pathlib import Path

import chess
import torch
from torch.utils.data import random_split

import processing
from net import ChessDataset, ChessNet

##init torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = Path(__file__).parent.parent
output = base_dir / "models"


def train_test_run(model_name="model", epochs=5):
    path = output / (model_name + ".pth")

    model = ChessNet()
    model.to(device)

    print(f"epochs: {epochs}")
    dataset = ChessDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    model.fit(train_ds, epochs)
    results = model.evaluate(test_ds)

    print("saving current model")
    torch.save(model.state_dict(), str(path))

    print("results from testing")
    for k in results.keys():
        print(f"{k}: {results[k]}")


def test_saved_model(model_name="model"):
    model = ChessNet()
    model.to(device)
    path = output / (model_name + ".pth")
    print("reading model")
    model.load_state_dict(torch.load(str(path), map_location=device))
    print("read model")

    dataset = ChessDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_ds = random_split(dataset, [train_size, test_size])

    print("testing for starting board")
    test_x = torch.from_numpy(processing.vectorise(chess.Board().fen()))
    test_x = test_x.to(device)

    model.eval()
    p_sigma, v_sigma = model.predict(test_x)

    k = 20  ##number of possible moves from init chess state
    k_probs, k_indecies = torch.topk(p_sigma, k)
    k_probs = k_probs / k_probs.sum()  # normalise

    print(
        f"policy over legal moves, {k_probs.squeeze(0)}\nindeces in normal sample pool: {k_indecies}"
    )
    print(f"game outcome: {v_sigma}")

    print("begning test")
    results = model.evaluate(test_ds)

    print("results from testing")
    for k in results.keys():
        print(f"{k}: {results[k]}")


if __name__ == "__main__":
    # train_test_run()
    test_saved_model()
