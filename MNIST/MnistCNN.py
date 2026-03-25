import torch.nn as nn

class MnistCNN(nn.Module):
    """
    Architecture
    ─────────────────────────────────────────────────
    Input  : (B, 1, 28, 28)
    Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
    Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
    Block 3: Conv(128)→ BN → ReLU → AdaptiveAvgPool
    Head   : Flatten → FC(256) → ReLU → Dropout → FC(10)
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 28x28 → 14x14
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(dropout),
            # Block 2 — 14x14 → 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(dropout),
            # Block 3 — 7x7 → 1x1
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
