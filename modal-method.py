import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Data
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

labels = np.array(train_dataset.targets)
secure_digits = [7, 8, 9]
public_idx = np.where(~np.isin(labels, secure_digits))[0]
secure_idx = np.where(np.isin(labels, secure_digits))[0]


public_dataset = Subset(train_dataset, public_idx)
secure_dataset = Subset(train_dataset, secure_idx)

public_loader = DataLoader(public_dataset, batch_size=128, shuffle=True, drop_last=True)
secure_loader = DataLoader(secure_dataset, batch_size=128, shuffle=True, drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# -------------------------
# Model
# -------------------------
class SecureMultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pub_branch = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.sec_branch = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.pub_dim = 32*7*7
        self.sec_dim = 16*7*7
        fused_dim = 256

        self.fusion = nn.Sequential(
            nn.Linear(self.pub_dim + self.sec_dim, fused_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(fused_dim, 10)

    def forward(self, x, secure_active: bool):
        h = self.stem(x)
        pub_feat = self.pub_branch(h)  # [B, pub_dim]

        if secure_active:
            sec_feat = self.sec_branch(h)  # [B, sec_dim]
        else:
            B = pub_feat.size(0)
            sec_feat = torch.zeros(B, self.sec_dim, device=pub_feat.device)

        fused = torch.cat([pub_feat, sec_feat], dim=1)
        fused = self.fusion(fused)
        logits = self.head(fused)
        return logits

model = SecureMultimodalNet().to(device)
criterion = nn.CrossEntropyLoss()

# Optimizers:
# We'll make *separate* optimizers for:
# - everything (full_optimizer) used on non-8 data
# - secure_only_optimizer used on 8 data (no stem/public)

full_optimizer = optim.Adam(
    list(model.stem.parameters()) +
    list(model.pub_branch.parameters()) +
    list(model.sec_branch.parameters()) +
    list(model.fusion.parameters()) +
    list(model.head.parameters()),
    lr=1e-3
)

secure_only_optimizer = optim.Adam(
    list(model.sec_branch.parameters()) +
    list(model.fusion.parameters()) +
    list(model.head.parameters()),
    lr=1e-3
)

@torch.no_grad()
def evaluate_per_class(model, loader, secure_active: bool):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, secure_active=secure_active)
        preds = logits.argmax(1)
        for digit in range(10):
            mask = (y == digit)
            if mask.any():
                correct[digit] += (preds[mask] == digit).sum().item()
                total[digit] += mask.sum().item()
    acc_per_digit = {d: (correct[d]/total[d] if total[d] > 0 else 0.0) for d in range(10)}
    overall = sum(correct.values()) / sum(total.values())
    return acc_per_digit, overall

def train_epoch(model, public_loader, secure_loader):
    model.train()
    pub_iter, sec_iter = iter(public_loader), iter(secure_loader)
    steps = min(len(public_loader), len(secure_loader))

    total_pub_loss = 0.0
    total_sec_loss = 0.0

    for _ in range(steps):
        # ---- Phase A: train on non-8 batch ----
        x_pub, y_pub = next(pub_iter)
        x_pub, y_pub = x_pub.to(device), y_pub.to(device)

        # full_optimizer updates EVERYTHING (including secure_branch!)
        # We train with secure_active=True so the head learns fused mode on non-8 data
        full_optimizer.zero_grad()
        logits_pub = model(x_pub, secure_active=True)
        loss_pub = criterion(logits_pub, y_pub)
        loss_pub.backward()
        full_optimizer.step()
        total_pub_loss += loss_pub.item()

        # Also OPTIONAL: teach model to handle secure_active=False mode on non-8,
        # so it knows how to behave when secure memory is revoked.
        full_optimizer.zero_grad()
        logits_pub_no_secure = model(x_pub, secure_active=False)
        loss_pub_no_secure = criterion(logits_pub_no_secure, y_pub)
        loss_pub_no_secure.backward()
        full_optimizer.step()
        total_pub_loss += loss_pub_no_secure.item()

        # ---- Phase B: train on 8 batch ----
        x_sec, y_sec = next(sec_iter)
        x_sec, y_sec = x_sec.to(device), y_sec.to(device)

        # secure_only_optimizer: NO stem/public updates.
        secure_only_optimizer.zero_grad()
        logits_sec = model(x_sec, secure_active=True)
        loss_sec = criterion(logits_sec, y_sec)
        loss_sec.backward()
        secure_only_optimizer.step()
        total_sec_loss += loss_sec.item()

    avg_pub_loss = total_pub_loss / (2*steps)  # we did two public updates per loop
    avg_sec_loss = total_sec_loss / steps
    return avg_pub_loss, avg_sec_loss

def train(model, epochs=5):
    for ep in range(1, epochs+1):
        loss_pub, loss_sec = train_epoch(model, public_loader, secure_loader)

        acc_on, overall_on = evaluate_per_class(model, test_loader, secure_active=True)
        acc_off, overall_off = evaluate_per_class(model, test_loader, secure_active=False)

        print(f"\nEpoch {ep}/{epochs}")
        print(f"  Public loss(avg): {loss_pub:.4f} | Secure loss(avg): {loss_sec:.4f}")
        print(f"  Overall (secure ON):  {overall_on:.3f}")
        print(f"  Overall (secure OFF): {overall_off:.3f}")
        print(f"  Per-digit accuracy (secure ON):")
        print("   " + " ".join([f"{d}:{acc_on[d]:.2f}" for d in range(10)]))
        print(f"  Per-digit accuracy (secure OFF):")
        print("   " + " ".join([f"{d}:{acc_off[d]:.2f}" for d in range(10)]))
        print("-"*60)

    print("\n=== FINAL REPORT ===")
    acc_on, overall_on = evaluate_per_class(model, test_loader, secure_active=True)
    acc_off, overall_off = evaluate_per_class(model, test_loader, secure_active=False)
    print(f"Overall accuracy (secure ON):  {overall_on:.4f}")
    print(f"Overall accuracy (secure OFF): {overall_off:.4f}")
    print("\nPer-digit accuracy WITH secure memory:")
    for d in range(10):
        print(f"  {d}: {acc_on[d]:.4f}")
    print("\nPer-digit accuracy WITHOUT secure memory:")
    for d in range(10):
        print(f"  {d}: {acc_off[d]:.4f}")
    print("==============================================================")

train(model, epochs=5)
