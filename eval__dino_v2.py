import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model

# === 設定 (請確認這裡的檔名是對的) ===
CSV_PATH = "isic-2024-challenge/train-metadata.csv"  # 確保路徑正確
IMAGE_FOLDER = "isic-2024-challenge/train-image/image"  # 確保路徑正確
MODEL_PATH = "dino_2080_epoch_10.pth"  # <--- 請改成你訓練出來的最後一個 .pth 檔名
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === pAUC 設定 ===
# ISIC 2024 Kaggle 主要指標：pAUC above 80% TPR（若你要用 88% TPR，改成 0.88 即可）
MIN_TPR = 0.80
OUTPUT_SCORES_CSV = "val_inference_scores.csv"  # 每筆推論分數輸出檔名


def compute_pauc_above_tpr(y_true, y_score, min_tpr: float = 0.80) -> float:
    """
    計算 ISIC 2024 所用的 pAUC-above-TPR：
      pAUC = ∫ max(TPR(FPR) - min_tpr, 0) dFPR
    最大值為 1 - min_tpr（例如 min_tpr=0.8，最大 0.2）

    作法：用 ROC 曲線找出 TPR 首次到達 min_tpr 的交點（線性內插），再用梯形法積分。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # 若只剩單一類別，roc_curve 會失敗；此時 pAUC 定義上可視為 0
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # tpr 單調不減；找第一個 >= min_tpr 的索引
    idx = np.searchsorted(tpr, min_tpr, side="left")
    if idx >= len(tpr):
        return 0.0  # 永遠達不到 min_tpr

    # 將「剛好跨過 min_tpr」的交點補進來，避免積分誤差
    if tpr[idx] == min_tpr:
        fpr_start = fpr[idx]
        fpr_seg = fpr[idx:]
        tpr_seg = tpr[idx:]
    else:
        if idx == 0:
            fpr_start = fpr[0]
        else:
            tpr1, tpr2 = tpr[idx - 1], tpr[idx]
            fpr1, fpr2 = fpr[idx - 1], fpr[idx]
            # 線性內插：tpr = tpr1 + w*(tpr2-tpr1) = min_tpr
            w = (min_tpr - tpr1) / (tpr2 - tpr1 + 1e-12)
            fpr_start = fpr1 + w * (fpr2 - fpr1)

        fpr_seg = np.concatenate([[fpr_start], fpr[idx:]])
        tpr_seg = np.concatenate([[min_tpr], tpr[idx:]])

    # 對 (tpr - min_tpr) 進行積分
    pauc = np.trapz(tpr_seg - min_tpr, fpr_seg)
    return float(max(pauc, 0.0))


# === 1. 必須重新定義模型架構 (跟訓練時一模一樣) ===
class SkinClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(768, 2)  # DINO 是二分類

    def forward(self, x):
        outputs = self.encoder(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


# === 2. 資料集類別 ===
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
        img_name = f"{isic_id}.jpg"
        path = os.path.join(self.img_dir, img_name)

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))  # 讀失敗給黑圖

        if self.transform:
            img = self.transform(img)

        return img, int(row["target"]), isic_id  # 回傳 isic_id 方便儲存每筆分數


def main():
    print(f"🚀 正在讀取資料... (使用模型: {MODEL_PATH})")

    # 讀取 CSV (加上 low_memory 防止警告)
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # 切分驗證集 (固定 random_state=42 確保跟訓練時切的一樣)
    _, val_df = train_test_split(
        df, test_size=0.1, stratify=df["target"], random_state=42
    )

    # === 採樣策略 (跟 EfficientNet 那邊一樣，取部分良性+全部惡性) ===
    val_pos = val_df[val_df["target"] == 1]
    # 取 5000 筆良性來測試 (太多跑很慢，太少不準)
    val_neg = val_df[val_df["target"] == 0].sample(n=5000, random_state=42)
    val_eval_df = (
        pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    )

    print(
        f"📊 評估樣本數: {len(val_eval_df)} (惡性: {len(val_pos)}, 良性: {len(val_neg)})"
    )

    # === 3. 載入 DINOv2 模型結構 ===
    print("🦖 重建 DINOv2 + LoRA 模型架構...")
    base_model = Dinov2Model.from_pretrained("facebook/dinov2-with-registers-base")

    # LoRA 設定 (必須跟訓練時完全一樣)
    peft_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1, bias="none"
    )
    base_model = get_peft_model(base_model, peft_config)

    # 套上分類頭
    model = SkinClassifier(base_model)

    # === 4. 載入權重 ===
    print(f"📥 載入權重檔案: {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print("❌ 找不到權重檔！請確認檔名是否正確。")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 權重載入成功！")
    except Exception as e:
        print(f"❌ 權重載入失敗，可能是架構不對稱。\n錯誤訊息: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # === 5. 開始預測 ===
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    loader = DataLoader(
        ISICDataset(val_eval_df, IMAGE_FOLDER, tfm),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    probs = []  # 存惡性的機率
    targs = []  # 存真實標籤
    ids = []  # 存 isic_id（每筆）

    print("🔍 開始推論...")
    with torch.no_grad():
        for x, y, batch_ids in tqdm(loader, desc="Evaluating"):
            x = x.to(DEVICE)
            out = model(x)  # 輸出是 [Batch, 2]

            # DINO 輸出兩欄，我們要用 Softmax 轉成機率，並取第 1 欄 (惡性機率)
            prob_malignant = torch.softmax(out, dim=1)[:, 1]

            probs.extend(prob_malignant.detach().cpu().numpy().tolist())
            targs.extend(y.numpy().tolist())
            ids.extend(list(batch_ids))

    # === 6. 儲存每筆分數 ===
    scores_df = pd.DataFrame(
        {"isic_id": ids, "target": targs, "prob_malignant": probs}
    )
    scores_df.to_csv(OUTPUT_SCORES_CSV, index=False)
    print(f"💾 已輸出每筆推論分數: {OUTPUT_SCORES_CSV}")

    # === 7. 產生報表 ===
    threshold = 0.5
    preds_bin = [1 if p > threshold else 0 for p in probs]

    print("\n" + "=" * 40)
    print(f"【 DINOv2 最終評估報告 (Threshold={threshold}) 】")

    # Full AUC
    try:
        auc = roc_auc_score(targs, probs)
        print(f"AUC Score: {auc:.4f}")
    except Exception:
        print("AUC Score: 無法計算 (可能是只有單一類別)")

    # pAUC (ISIC 2024)
    pauc = compute_pauc_above_tpr(targs, probs, min_tpr=MIN_TPR)
    print(f"pAUC-above-TPR (min_tpr={MIN_TPR:.2f}): {pauc:.6f}  (max={1-MIN_TPR:.2f})")
    print(f"pAUC normalized (pAUC/(1-min_tpr)): {pauc / max(1e-12, (1 - MIN_TPR)):.6f}")

    print(f"Accuracy:  {accuracy_score(targs, preds_bin):.4f}")
    print("-" * 40)
    print(classification_report(targs, preds_bin, target_names=["Benign", "Malignant"]))

    # 顯示混淆矩陣
    cm = confusion_matrix(targs, preds_bin)
    print("混淆矩陣 (Confusion Matrix):")
    print(cm)
    print(f"\n[良性預測對: {cm[0][0]}]  [誤判為惡性: {cm[0][1]}]")
    print(f"[誤判為良性: {cm[1][0]}]  [惡性預測對: {cm[1][1]}] <--- 重點看這裡！")
    print("=" * 40)


if __name__ == "__main__":
    main()
