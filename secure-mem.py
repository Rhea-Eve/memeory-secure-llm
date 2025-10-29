# ==============================================================
# Secure-Memory CNN Experiment (digit 8 as secure information)
# ==============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# --------------------------------------------------------------
# 1. Setup & dataset
# --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split into public (non-8) and secure (8) subsets
labels = np.array(train_dataset.targets)
public_indices = np.where(labels != 8)[0]
secure_indices = np.where(labels == 8)[0]

public_dataset = Subset(train_dataset, public_indices)
secure_dataset = Subset(train_dataset, secure_indices)

public_loader = DataLoader(public_dataset, batch_size=128, shuffle=True, drop_last=True)
secure_loader = DataLoader(secure_dataset, batch_size=128, shuffle=True, drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# --------------------------------------------------------------
# 2. Model definition
# --------------------------------------------------------------
class SecureSplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        # shared feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # public branch: learns 0–7,9
        self.pub_branch = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.pub_head = nn.Linear(32*7*7, 9)
        # secure branch: learns only 8
        self.sec_branch = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.sec_head = nn.Linear(16*7*7, 1)

    def forward_public_logits(self, x):
        h = self.stem(x)
        h_pub = self.pub_branch(h)
        return self.pub_head(h_pub)

    def forward_secure_logit(self, x):
        h = self.stem(x)
        h_sec = self.sec_branch(h)
        return self.sec_head(h_sec).squeeze(1)

# --------------------------------------------------------------
# 3. Label remapping helper
# --------------------------------------------------------------
remap_dict = {i:i for i in range(8)}
remap_dict[9] = 8
def remap_public_labels(y):
    mapped = [remap_dict[int(v)] for v in y.tolist()]
    return torch.tensor(mapped, device=y.device, dtype=torch.long)

# --------------------------------------------------------------
# 4. Training setup
# --------------------------------------------------------------
model = SecureSplitNet().to(device)
criterion_pub = nn.CrossEntropyLoss()
criterion_sec = nn.BCEWithLogitsLoss()

optimizer_public = optim.Adam(
    list(model.stem.parameters()) +
    list(model.pub_branch.parameters()) +
    list(model.pub_head.parameters()),
    lr=1e-3
)
optimizer_secure = optim.Adam(
    list(model.sec_branch.parameters()) +
    list(model.sec_head.parameters()),
    lr=1e-3
)

# --------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------
def train_epoch(model, public_loader, secure_loader):
    model.train()
    pub_iter, sec_iter = iter(public_loader), iter(secure_loader)
    steps = min(len(public_loader), len(secure_loader))

    for _ in range(steps):
        # --- Public step (non-8) ---
        x_pub, y_pub = next(pub_iter)
        x_pub, y_pub = x_pub.to(device), y_pub.to(device)
        mask = (y_pub != 8)
        x_pub, y_pub = x_pub[mask], y_pub[mask]
        y_pub_remap = remap_public_labels(y_pub)

        optimizer_public.zero_grad()
        logits_pub = model.forward_public_logits(x_pub)
        loss_pub = criterion_pub(logits_pub, y_pub_remap)
        loss_pub.backward()
        optimizer_public.step()

        # --- Secure step (8 only) ---
        x_sec, y_sec = next(sec_iter)
        x_sec, y_sec = x_sec.to(device), y_sec.to(device)
        y_is8 = torch.ones_like(y_sec, dtype=torch.float32, device=device)

        optimizer_secure.zero_grad()
        logit_sec = model.forward_secure_logit(x_sec)
        loss_sec = criterion_sec(logit_sec, y_is8)
        loss_sec.backward()
        optimizer_secure.step()

def train(model, epochs=5):
    for ep in range(epochs):
        train_epoch(model, public_loader, secure_loader)
        print(f"Epoch {ep+1}/{epochs} complete")

# --------------------------------------------------------------
# 6. Evaluation helpers
# --------------------------------------------------------------
@torch.no_grad()
def fused_logits(model, x):
    model.eval()
    pub_logits = model.forward_public_logits(x)
    sec_logit  = model.forward_secure_logit(x)
    B = x.size(0)
    full_logits = torch.zeros(B, 10, device=device)
    full_logits[:,0:8] = pub_logits[:,0:8]
    full_logits[:,8]   = sec_logit
    full_logits[:,9]   = pub_logits[:,8]
    return full_logits

@torch.no_grad()
def fused_logits_no_secure(model, x):
    model.eval()
    pub_logits = model.forward_public_logits(x)
    B = x.size(0)
    full_logits = torch.zeros(B, 10, device=device)
    full_logits[:,0:8] = pub_logits[:,0:8]
    full_logits[:,8]   = -1e9   # disable secure branch
    full_logits[:,9]   = pub_logits[:,8]
    return full_logits

@torch.no_grad()
def evaluate(model, loader):
    correct_full = correct_no = total = 0
    correct8_full = correct8_no = tot8 = 0
    correctO_full = correctO_no = totO = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits_full = fused_logits(model, x)
        logits_no   = fused_logits_no_secure(model, x)
        pred_full = logits_full.argmax(1)
        pred_no   = logits_no.argmax(1)

        correct_full += (pred_full == y).sum().item()
        correct_no   += (pred_no == y).sum().item()
        total += y.size(0)

        mask8 = (y == 8)
        if mask8.any():
            correct8_full += (pred_full[mask8] == 8).sum().item()
            correct8_no   += (pred_no[mask8] == 8).sum().item()
            tot8 += mask8.sum().item()
        maskO = ~mask8
        if maskO.any():
            correctO_full += (pred_full[maskO] == y[maskO]).sum().item()
            correctO_no   += (pred_no[maskO] == y[maskO]).sum().item()
            totO += maskO.sum().item()

    return {
        "overall_full": correct_full/total,
        "overall_no_secure": correct_no/total,
        "acc_8_full": correct8_full/max(tot8,1),
        "acc_8_no_secure": correct8_no/max(tot8,1),
        "acc_not8_full": correctO_full/max(totO,1),
        "acc_not8_no_secure": correctO_no/max(totO,1)
    }

# --------------------------------------------------------------
# 7. Run experiment
# --------------------------------------------------------------
train(model, epochs=5)
results = evaluate(model, test_loader)
print("Results:", results)
