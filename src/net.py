import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, Dataset

base_dir = Path(__file__).parent.parent
PROCESSED_PATH = base_dir / "datasets" / "processed" / "chess_dataset.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))


class ChessDataset(Dataset):
    def __init__(self, npz_file=PROCESSED_PATH):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data["states"]).float()
        self.yp = torch.from_numpy(data["policy"]).long()
        self.yv = torch.from_numpy(data["values"]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, yp, yv = self.X[idx], self.yp[idx], self.yv[idx]
        return x, yp, yv


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_POSSIBLE_MOVES = 8 * 8 * 73

        self.trunk = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, self.N_POSSIBLE_MOVES),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

    def forward(self, x):
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def predict(self, x):
        self.eval()
        x = x.unsqueeze(0)
        policy, value = self(x)
        p_sigma = F.softmax(policy, dim=1)
        return p_sigma, value

    def fit(self, train_dataset, epochs=5):
        LAMBDA = 1.0  # loss ofset
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        start = time.time()
        for i in range(epochs):
            self.train()
            print(f"Starting Epoch {i+1}")
            for X, yp, yv in train_loader:
                X, yp, yv = (
                    X.to(device),
                    yp.to(device),
                    yv.to(device),
                )
                self.optimizer.zero_grad()

                policy_logits, value = self(X)

                p_loss = self.policy_loss(policy_logits, yp)
                v_loss = self.value_loss(value.squeeze(), yv)
                loss = p_loss + LAMBDA * v_loss
                loss.backward()
                self.optimizer.step()

        print(f"training took: {start - time.time()}")

    def evaluate(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        p_correct = 0
        p_topk_correct = 0
        v_sign_correct = 0
        v_mse_sum = 0

        total = 0

        with torch.no_grad():
            for X, yp, yv in test_loader:
                X, yp, yv = (
                    X.to(device),
                    yp.to(device),
                    yv.to(device),
                )

                p_logits, value_pred = self(X)
                value_pred = value_pred.squeeze(1)

                # policy top-1
                preds = p_logits.argmax(dim=1)
                p_correct += (preds == yp).sum().item()

                # policy topk
                topk_preds = p_logits.topk(5, dim=1).indices
                p_topk_correct += (
                    (topk_preds == yp.unsqueeze(1)).any(dim=1).sum().item()
                )

                v_mse_sum += F.mse_loss(value_pred, yv, reduction="sum").item()

                v_sign_correct += (
                    (torch.sign(value_pred) == torch.sign(yv)).sum().item()
                )

                total += yp.size(0)

        return {
            "policy_top1": 100 * p_correct / total,
            "policy_topk": 100 * p_topk_correct / total,
            "value_mse": 100 * v_mse_sum / total,
            "value_sign_acc": 100 * v_sign_correct / total,
        }
