import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm  # ✅ 進度條

# =========================
# 0) Config（依你指定）
# =========================
CSV_PATH   = "D:/Jeff/save/b2025/course/AI/isic-2024-challenge/train-metadata.csv"
IMAGE_DIR  = "D:/Jeff/save/b2025/course/AI/isic-2024-challenge/train-image/image"  # {isic_id}.jpg

CFG = {
    "img_size": 224,
    "unet_base": 64,
    "batch_size": 128,         # 224 分類通常 OK；若 OOM 改 16
    "num_workers": 8,
    "steps_per_epoch": 750,  # epoch_len = batch_size * steps_per_epoch
    "epochs": 10,
    "lr": 8e-4,
    "seed": 42,
    "use_amp": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 1) 讀圖檢查
# =========================
def check_image_loading(df, img_dir, n=3):
    print(f"[{time.strftime('%H:%M:%S')}] 檢查圖片讀取...")
    sample_ids = df.sample(n, random_state=CFG["seed"])["isic_id"].values
    for isic_id in sample_ids:
        path = os.path.join(img_dir, f"{isic_id}.jpg")
        Image.open(path).convert("RGB")
    print("✅ 圖片讀取路徑檢查通過")

# =========================
# 2) Dataset
# =========================
class ISIC2024ClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        isic_id = row["isic_id"]
        y = float(row["target"])

        path = os.path.join(self.img_dir, f"{isic_id}.jpg")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.float32)

# =========================
# 3) Model
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetEncoderClassifier(nn.Module):
    def __init__(self, in_ch=3, base=64, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.bottleneck = ConvBlock(base*8, base*16)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base*16, 1)

    def forward(self, x):
        x = self.enc1(x); x = self.pool(x)
        x = self.enc2(x); x = self.pool(x)
        x = self.enc3(x); x = self.pool(x)
        x = self.enc4(x); x = self.pool(x)
        x = self.bottleneck(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop(x)
        logit = self.fc(x).squeeze(1)
        return logit

# =========================
# 4) DataLoaders + sampler
# =========================
def build_loaders():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    required_cols = {"isic_id", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train-metadata.csv 缺少必要欄位：{missing}")

    check_image_loading(df, IMAGE_DIR, n=3)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df["target"],
        random_state=CFG["seed"],
    )

    train_tfm = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    count_pos = int((train_df["target"] == 1).sum())
    count_neg = int((train_df["target"] == 0).sum())
    w_pos = 1.0 / max(count_pos, 1)
    w_neg = 1.0 / max(count_neg, 1)

    sample_weights = np.where(train_df["target"].values == 1, w_pos, w_neg)
    sample_weights = torch.from_numpy(sample_weights).double()

    epoch_len = CFG["batch_size"] * CFG["steps_per_epoch"]
    sampler = WeightedRandomSampler(sample_weights, num_samples=epoch_len, replacement=True)

    train_ds = ISIC2024ClsDataset(train_df, IMAGE_DIR, transform=train_tfm)
    val_ds   = ISIC2024ClsDataset(val_df,   IMAGE_DIR, transform=val_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        persistent_workers=(CFG["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        persistent_workers=(CFG["num_workers"] > 0),
    )

    print(f"Train pool: {len(train_df)} | Val: {len(val_df)} | epoch_len(samples): {epoch_len}")
    print(f"Pos/Neg in train pool: {count_pos}/{count_neg} | w_pos/w_neg: {w_pos:.6e}/{w_neg:.6e}")
    return train_loader, val_loader

# =========================
# 5) Train / Eval with tqdm
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        pbar.set_postfix({"loss": f"{(total_loss/max(total_n,1)):.4f}"})

    return total_loss / max(total_n, 1)

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # moving average loss
        pbar.set_postfix({"loss": f"{(total_loss/max(total_n,1)):.4f}"})

    return total_loss / max(total_n, 1)

def main():
    seed_everything(CFG["seed"])
    device = CFG["device"]

    train_loader, val_loader = build_loaders()

    model = UNetEncoderClassifier(in_ch=3, base=CFG["unet_base"], dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2)

    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["use_amp"] and device.startswith("cuda")))
    scaler_obj = scaler if scaler.is_enabled() else None

    best_val = float("inf")
    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n===== Epoch {epoch:02d}/{CFG['epochs']} =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler_obj)
        va_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "cfg": CFG}, "best_unet_encoder_cls.pth")
            print("  ↳ saved: best_unet_encoder_cls.pth")

if __name__ == "__main__":
    main()
