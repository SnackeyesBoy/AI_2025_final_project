import os
import torch
import pandas as pd
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, accuracy_score, fbeta_score,
    roc_curve, auc
)
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. 設定區域
# ==========================================
CSV_PATH = "isic-2024-challenge/train-metadata.csv"
IMAGE_FOLDER = "isic-2024-challenge/train-image/image"
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你要輸出的逐筆分數檔
PRED_SAVE_PATH = "val_pred_scores.csv"

# ISIC2024: pAUC above 80% TPR
MIN_TPR = 0.80


# ==========================================
# 2. pAUC 計算（TPR >= min_tpr 的部分面積）
# 最大值為 (1 - min_tpr) * 1，因此 MIN_TPR=0.8 時滿分 0.2
# ==========================================
from isic_pauc import isic_pauc_above_tpr


# ==========================================
# 3. 資料集類別
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
        isic_id = row["isic_id"]
        path = os.path.join(self.img_dir, f"{isic_id}.jpg")

        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        target = torch.tensor(row["target"], dtype=torch.float)
        return img, target, isic_id


# ==========================================
# 4. 主程式
# ==========================================
def main():
    print("正在讀取並準備評估資料...")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except:
        print("❌ 錯誤：找不到 train-metadata.csv")
        return

    # 切分驗證集 (保持與訓練一致)
    _, val_df = train_test_split(df, test_size=0.1, stratify=df["target"], random_state=42)

    # === 採樣策略 ===
    # 取出所有惡性 + 5000 張良性
    val_pos = val_df[val_df["target"] == 1]
    val_neg = val_df[val_df["target"] == 0].sample(n=5000, random_state=42)
    val_eval_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"評估樣本總數: {len(val_eval_df)}")
    print(f" -> 包含惡性樣本: {len(val_pos)} (全部納入)")
    print(f" -> 包含良性樣本: {len(val_neg)}")

    # 載入模型
    print("載入模型與權重...")
    model = timm.create_model("tf_efficientnetv2_b0", pretrained=False, num_classes=1)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 成功載入 best_model.pth")
    else:
        print("❌ 錯誤：找不到 best_model.pth")
        return

    model.to(DEVICE)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        ISICDataset(val_eval_df, IMAGE_FOLDER, tfm),
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # 預測（並保留逐筆 isic_id）
    preds, targs, ids = [], [], []
    with torch.no_grad():
        for x, y, isic_id in tqdm(loader, desc="正在評估模型"):
            x = x.to(DEVICE)
            out = model(x)  # (B,1)
            score = torch.sigmoid(out).squeeze(1).detach().cpu().numpy()  # (B,)
            preds.extend(score.tolist())
            targs.extend(y.numpy().tolist())
            ids.extend(list(isic_id))

    # === 存檔：逐筆分數 ===
    pred_df = pd.DataFrame({
        "isic_id": ids,
        "target": targs,
        "score": preds
    })
    pred_df.to_csv(PRED_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已輸出逐筆分數到: {PRED_SAVE_PATH}")

    # === 計算 ISIC2024 pAUC (TPR>=MIN_TPR) ===
    raw_pauc = isic_pauc_above_tpr(targs, preds, min_tpr=MIN_TPR, normalize=False)
    norm_pauc = isic_pauc_above_tpr(targs, preds, min_tpr=MIN_TPR, normalize=True)
# === 原本的最佳門檻搜尋 (F2) ===
    print("\n正在尋找最佳分類門檻 (以 F2-Score 為基準，優先抓出癌症)...")
    best_score = 0
    best_thresh = 0.5

    for thresh in np.arange(0.001, 0.50, 0.001):
        preds_bin = [1 if p > thresh else 0 for p in preds]
        score = fbeta_score(targs, preds_bin, beta=2)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"🏆 最佳門檻值已鎖定: {best_thresh:.4f}")

    # === 生成最終報表 ===
    final_preds_bin = [1 if p > best_thresh else 0 for p in preds]

    cm = confusion_matrix(targs, final_preds_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print("\n" + "=" * 50)
    print("【 最終完美版報告 (請截圖這張放 PPT) 】")
    print("=" * 50)
    print(f"最佳門檻值 (Threshold): {best_thresh:.4f}")
    print(f"AUC (鑑別力):          {roc_auc_score(targs, preds):.4f}")
    print(f"pAUC (TPR >= {MIN_TPR:.2f}, raw):        {raw_pauc:.6f}")
    print(f"pAUC (TPR >= {MIN_TPR:.2f}, normalized): {norm_pauc:.6f}")
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


if __name__ == "__main__":
    main()