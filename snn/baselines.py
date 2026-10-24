"""
Standard (non-spiking) baselines for comparison: a small CNN and a small
LSTM, both operating on the same MFCC frames the SNN sees (just without the
spike encoding step -- these consume continuous-valued MFCCs directly).

Kept deliberately small to be a fair-ish parameter-count comparison against
the SNN, and to train quickly on CPU/Kaggle GPU alike.
"""

import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, n_mfcc: int, n_time_steps: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, time, n_mfcc) -> conv wants (batch, channels=n_mfcc, time)
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.classifier(x)


class LSTMBaseline(nn.Module):
    def __init__(self, n_mfcc: int, n_classes: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mfcc, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])
