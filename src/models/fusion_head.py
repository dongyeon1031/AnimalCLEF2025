import torch, torch.nn as nn, torch.nn.functional as F

class FusionMLP(nn.Module):
    """Concat(Mega 1536, EVA02 768, |diff| 2304) → MLP → prob."""
    def __init__(self, in_dim=2304):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, feat_q, feat_db):
        x = torch.cat([feat_q, feat_db, torch.abs(feat_q - feat_db)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))