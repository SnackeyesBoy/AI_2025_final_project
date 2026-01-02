import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

from isic_pauc import isic_pauc_above_tpr

# ISIC2024: pAUC above 80% TPR6
MIN_TPR = 0.80

# ==========================================
# 1. 設定區域（你自行改路徑）
# ==========================================
CSV_PATH   = "D:/Jeff/save/b2025/course/AI/isic-2024-challenge/train-metadata.csv"
IMAGE_FOLDER  = "D:/Jeff/save/b2025/course/AI/isic-2024-challenge/train-image/image"
MODEL_PATH = "best_unet_encoder_cls.pth"  # 你的 U-Net encoder classifier 權重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
UNET_BASE = 64
BATCH_SIZE = 64
NUM_WORKERS = 0
RANDOM_STATE = 42

# ==========================================
# 2. 資料集類別（沿用 eval (1).py 的讀取風格）
# ==========================================
class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['isic_id']}.jpg")
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))  # 讀不到給黑圖（與參考程式一致）
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["target"], dtype=torch.float32), row["isic_id"]


# ==========================================
# 3. U-Net encoder classifier（不輸出 mask）
# ==========================================
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
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 16, 1)

    def forward(self, x):
        x = self.enc1(x); x = self.pool(x)
        x = self.enc2(x); x = self.pool(x)
        x = self.enc3(x); x = self.pool(x)
        x = self.enc4(x); x = self.pool(x)
        x = self.bottleneck(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop(x)
        logit = self.fc(x).squeeze(1)   # (B,)
        return logit

# ==========================================
# 4. 權重載入（支援兩種格式）
#    - state_dict 直接存
#    - 或 {"model": state_dict, "cfg": ...}
# ==========================================
def load_checkpoint(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型權重：{model_path}")

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    return model

# ==========================================
# 5. 主程式（流程對齊 eval (1).py）
# ==========================================
def main():
    print("正在讀取並準備評估資料...")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except:
        print("❌ 錯誤：找不到 train-metadata.csv")
        return

    # 切分出驗證集（與參考程式一致）
    _, val_df = train_test_split(
        df, test_size=0.1, stratify=df["target"], random_state=RANDOM_STATE
    )

    # 評估抽樣：全部惡性 + 隨機 5000 良性（與參考程式一致）
    val_pos = val_df[val_df["target"] == 1]
    val_neg_n = min(5000, len(val_df[val_df["target"] == 0]))
    val_neg = val_df[val_df["target"] == 0].sample(n=val_neg_n, random_state=RANDOM_STATE)

    val_eval_df = (
        pd.concat([val_pos, val_neg])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(f"評估樣本總數: {len(val_eval_df)}")
    print(f" -> 包含惡性樣本: {len(val_pos)} (全部納入)")
    print(f" -> 包含良性樣本: {len(val_neg)} (隨機抽樣)")

    print("載入模型與權重...")
    model = UNetEncoderClassifier(in_ch=3, base=UNET_BASE, dropout=0.2)
    model = load_checkpoint(model, MODEL_PATH, DEVICE)
    model.to(DEVICE)
    model.eval()
    print("✅ 成功載入模型權重")

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(
        ISICDataset(val_eval_df, IMAGE_FOLDER, tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    preds, targs, ids = [], [], []

    with torch.no_grad():
        for x, y, isic_id in tqdm(loader, desc="正在評估 U-Net 分類模型"):
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()

            preds.extend(prob.tolist())
            targs.extend(y.numpy().tolist())
            ids.extend(isic_id)


    
    preds = np.array(preds, dtype=np.float64)
    targs = np.array(targs, dtype=np.int64)

    # ===============================
    # 儲存所有推論分數
    # ===============================
    out_df = pd.DataFrame({
        "isic_id": ids,
        "target": targs,
        "pred_score": preds
    })

    out_path = "unet_val_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ 已儲存所有推論分數至: {out_path}")

    # === 自動尋找最佳門檻 (Threshold Tuning) ===
    print("\n正在尋找最佳分類門檻 (以 F1-Score 為基準)...")
    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in np.arange(0.01, 0.60, 0.01):
        preds_bin = (preds > thresh).astype(int)
        f1 = f1_score(targs, preds_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    print(f"🏆 最佳門檻值已鎖定: {best_thresh:.2f} | Best F1={best_f1:.4f}")

    final_preds_bin = (preds > best_thresh).astype(int)

    # 混淆矩陣
    cm = confusion_matrix(targs, final_preds_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # 報表（與參考程式一致的輸出風格）
    print("\n" + "=" * 50)
    print("【 最終報告 】")
    print("=" * 50)
    print(f"最佳門檻值 (Threshold): {best_thresh:.2f}")

    # AUC：需要同時存在 0/1 才能算
    if len(np.unique(targs)) == 2:
        print(f"AUC (鑑別力):          {roc_auc_score(targs, preds):.4f}")
    else:
        print("AUC (鑑別力):          N/A（驗證資料只有單一類別）")

    print(f"Accuracy (準確率):     {accuracy_score(targs, final_preds_bin):.4f}")
    print("-" * 50)

    print("混淆矩陣 (Confusion Matrix):")
    print(f"[[{tn}\t{fp}]")
    print(f" [{fn}\t{tp}]]\n")
    print(f"[良性預測對: {tn}]  [誤判為惡性: {fp}]")
    print(f"[誤判為良性: {fn}]  [惡性預測對: {tp}] <--- 重點看這裡！(Recall高)")

    print("-" * 50)
    print("詳細分類指標:")
    print(classification_report(targs, final_preds_bin, target_names=["Benign", "Malignant"]))
    print("=" * 50)
    # ===============================
    # pAUC（ISIC 2024 核心指標）
    # ===============================
    if len(np.unique(targs)) == 2:
        raw_pauc = isic_pauc_above_tpr(targs, preds, min_tpr=MIN_TPR, normalize=False)
        norm_pauc = isic_pauc_above_tpr(targs, preds, min_tpr=MIN_TPR, normalize=True)

        print(f"pAUC (TPR >= {MIN_TPR:.2f}, raw):        {raw_pauc:.6f}")
        print(f"pAUC (TPR >= {MIN_TPR:.2f}, normalized): {norm_pauc:.6f}")
    else:
        print("pAUC: N/A（驗證資料只有單一類別）")

if __name__ == "__main__":
    main()
