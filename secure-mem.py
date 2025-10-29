# ==============================================================
# Secure-Memory CNN Experiment — Improved Metrics + Reporting
# ==============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------------------
# 1. Dataset setup
# --------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

labels = np.array(train_dataset.targets)
public_idx = np.where(labels != 8)[0]
secure_idx = np.where(labels == 8)[0]

public_dataset = Subset(train_dataset, public_idx)
secure_dataset = Subset(train_dataset, secure_idx)

public_loader = DataLoader(public_dataset, batch_size=128, shuffle=True, drop_last=True)
secure_loader = DataLoader(secure_dataset, batch_size=128, shuffle=True, drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# --------------------------------------------------------------
# 2. Model definition
# --------------------------------------------------------------
class SecureSplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pub_branch = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.pub_head = nn.Linear(32*7*7, 9)
        self.sec_branch = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.sec_head = nn.Linear(16*7*7, 1)

    def forward_public_logits(self, x):
        h = self.stem(x)
        return self.pub_head(self.pub_branch(h))

    def forward_secure_logit(self, x):
        h = self.stem(x)
        return self.sec_head(self.sec_branch(h)).squeeze(1)

# --------------------------------------------------------------
# 3. Label remap for public branch
# --------------------------------------------------------------
remap = {i:i for i in range(8)}
remap[9] = 8
def remap_public_labels(y):
    mapped = [remap[int(v)] for v in y.tolist()]
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
# 5. Fusion helpers for evaluation
# --------------------------------------------------------------
@torch.no_grad()
def fused_logits(model, x):
    model.eval()
    pub_logits = model.forward_public_logits(x)
    sec_logit  = model.forward_secure_logit(x)
    B = x.size(0)
    full = torch.zeros(B, 10, device=device)
    full[:,0:8] = pub_logits[:,0:8]
    full[:,8]   = sec_logit
    full[:,9]   = pub_logits[:,8]
    return full

@torch.no_grad()
def fused_logits_no_secure(model, x):
    model.eval()
    pub_logits = model.forward_public_logits(x)
    B = x.size(0)
    full = torch.zeros(B, 10, device=device)
    full[:,0:8] = pub_logits[:,0:8]
    full[:,8]   = -1e9
    full[:,9]   = pub_logits[:,8]
    return full

# --------------------------------------------------------------
# 6. Per-digit evaluation
# --------------------------------------------------------------
@torch.no_grad()
def evaluate_per_class(model, loader, secure=True):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = fused_logits(model, x) if secure else fused_logits_no_secure(model, x)
        preds = logits.argmax(1)
        for i in range(10):
            mask = (y == i)
            if mask.any():
                correct[i] += (preds[mask] == i).sum().item()
                total[i] += mask.sum().item()
    acc = {i: (correct[i] / total[i]) if total[i]>0 else 0.0 for i in range(10)}
    overall = sum(correct.values()) / sum(total.values())
    return acc, overall

# --------------------------------------------------------------
# 7. Training loop with stats
# --------------------------------------------------------------
def train_epoch(model, pub_loader, sec_loader):
    model.train()
    pub_iter, sec_iter = iter(pub_loader), iter(sec_loader)
    steps = min(len(pub_loader), len(sec_loader))
    total_pub_loss, total_sec_loss = 0, 0
    for _ in range(steps):
        # ---- Public ----
        x_pub, y_pub = next(pub_iter)
        x_pub, y_pub = x_pub.to(device), y_pub.to(device)
        y_pub_remap = remap_public_labels(y_pub[y_pub!=8])
        optimizer_public.zero_grad()
        loss_pub = criterion_pub(model.forward_public_logits(x_pub[y_pub!=8]), y_pub_remap)
        loss_pub.backward(); optimizer_public.step()
        total_pub_loss += loss_pub.item()

        # ---- Secure ----
        x_sec, y_sec = next(sec_iter)
        x_sec, y_sec = x_sec.to(device), y_sec.to(device)
        y_is8 = torch.ones_like(y_sec, dtype=torch.float32, device=device)
        optimizer_secure.zero_grad()
        loss_sec = criterion_sec(model.forward_secure_logit(x_sec), y_is8)
        loss_sec.backward(); optimizer_secure.step()
        total_sec_loss += loss_sec.item()
    return total_pub_loss/steps, total_sec_loss/steps

# --------------------------------------------------------------
# 8. Training driver
# --------------------------------------------------------------
def train(model, epochs=5):
    for ep in range(1, epochs+1):
        loss_pub, loss_sec = train_epoch(model, public_loader, secure_loader)
        acc_secure, overall_secure = evaluate_per_class(model, test_loader, secure=True)
        acc_no, overall_no = evaluate_per_class(model, test_loader, secure=False)

        print(f"\nEpoch {ep}/{epochs}")
        print(f"  Public Loss: {loss_pub:.4f} | Secure Loss: {loss_sec:.4f}")
        print(f"  Overall Acc (secure on):  {overall_secure:.3f}")
        print(f"  Overall Acc (secure off): {overall_no:.3f}")
        print("  Per-digit Acc (secure on): ")
        for i in range(10):
            print(f"    {i}: {acc_secure[i]:.3f}", end=" | ")
        print("\n  Per-digit Acc (secure off): ")
        for i in range(10):
            print(f"    {i}: {acc_no[i]:.3f}", end=" | ")
        print("\n" + "-"*60)

# --------------------------------------------------------------
# 9. Run experiment
# --------------------------------------------------------------
train(model, epochs=5)

# Final report
print("\n=== FINAL REPORT ===")
acc_full, overall_full = evaluate_per_class(model, test_loader, secure=True)
acc_nosec, overall_nosec = evaluate_per_class(model, test_loader, secure=False)

print(f"\nOverall (secure ON):  {overall_full:.4f}")
print(f"Overall (secure OFF): {overall_nosec:.4f}\n")

print("Per-digit accuracy WITH secure memory:")
for i in range(10):
    print(f"  {i}: {acc_full[i]:.4f}")
print("\nPer-digit accuracy WITHOUT secure memory:")
for i in range(10):
    print(f"  {i}: {acc_nosec[i]:.4f}")
print("==============================================================")
