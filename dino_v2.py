import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 1. 超參數設定 =================
# 資料路徑設定 (指向同目錄下的資料夾)
DATA_ROOT = "isic-2024-challenge" 
CSV_PATH = os.path.join(DATA_ROOT, "train-metadata.csv")
IMG_FOLDER = os.path.join(DATA_ROOT, "train-image/image")

# 訓練參數 (針對 RTX 2080 Super 優化)
BATCH_SIZE = 16            # 8GB VRAM 的安全值
EPOCHS = 10                # 訓練輪數
SAMPLES_PER_EPOCH = 30000  # 【黃金策略】每一輪只看 3 萬張，但從全量資料隨機抽
LR = 1e-4                  # 學習率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 定義 Dataset 與 Model Class =================

class ISICLazyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.data = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['isic_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        label = int(row['target'])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform: image = self.transform(image)
            return image, label
        except Exception as e:
            # 讀取失敗回傳黑圖，避免崩潰
            return torch.zeros(3, 224, 224), label

class SkinClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(768, 2) # 二分類 (良性/惡性)

    def forward(self, x):
        outputs = self.encoder(x)
        # 取 CLS Token (第 0 個向量)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# ================= 3. 主執行區塊 (Windows 保護鎖) =================
if __name__ == '__main__':
    print(f"🚀 [1/7] 正在初始化設定...")
    print(f"   使用裝置: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # --- 路徑檢查 ---
    if not os.path.exists(CSV_PATH):
        print(f"❌ 錯誤：找不到 CSV 檔！路徑: {os.path.abspath(CSV_PATH)}")
        exit()
    if not os.path.exists(IMG_FOLDER):
        print(f"❌ 錯誤：找不到圖片資料夾！路徑: {os.path.abspath(IMG_FOLDER)}")
        exit()

    # --- 讀取數據 ---
    print("📂 [2/7] 讀取 CSV 標籤檔...")
    # low_memory=False 消除 DtypeWarning
    df_full = pd.read_csv(CSV_PATH, low_memory=False) 
    df_full = df_full[['isic_id', 'target']]

    # 切分訓練集與驗證集 (固定 random_state 以確保公平比較)
    train_df = df_full.sample(frac=0.8, random_state=42)
    val_df = df_full.drop(train_df.index)

    # --- 影像前處理 ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ISICLazyDataset(train_df, IMG_FOLDER, transform=train_transform)
    val_dataset = ISICLazyDataset(val_df, IMG_FOLDER, transform=val_transform)

    # --- 【關鍵】平衡採樣策略 ---
    print("⚖️  [3/7] 計算類別權重 (全量池 + 動態採樣)...")
    targets = train_df['target'].values
    class_counts = [(targets == 0).sum(), (targets == 1).sum()]
    if class_counts[1] == 0: class_counts[1] = 1 # 防呆
    weight = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weight = torch.tensor([weight[t] for t in targets])

    # replacement=True 代表允許重複抽樣，保證每一輪 3 萬張都能看到不同的良性圖片
    sampler = WeightedRandomSampler(samples_weight, num_samples=SAMPLES_PER_EPOCH, replacement=True)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

    # --- 視覺驗證：檢查是否有讀到圖片 ---
    print("👀 [4/7] 正在進行圖片讀取檢查 (存檔 debug_check.png)...")
    try:
        check_iter = iter(train_loader)
        images, labels = next(check_iter)
        # 反向標準化以便肉眼觀察
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        check_imgs = [inv_normalize(img) for img in images]
        save_image(torch.stack(check_imgs), "debug_check.png", nrow=4)
        print("   ✅ 檢查成功！請打開 'debug_check.png' 確認圖片是否正常。")
    except Exception as e:
        print(f"   ⚠️ 讀取檢查失敗 (可能是路徑問題): {e}")

    # --- 載入模型 ---
    print("🦖 [5/7] 載入 DINOv2 (With Registers) + LoRA...")
    base_model = Dinov2Model.from_pretrained("facebook/dinov2-with-registers-base")

    # LoRA 設定：只微調 Attention (query, value)，極省顯存
    peft_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["query", "value"], 
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    model = SkinClassifier(model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True) # FP16 混合精度

    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    # --- 開始訓練 ---
    print(f"🔥 [6/7] 開始戰鬥！預計訓練 {EPOCHS} 輪...")
    print("-" * 50)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if i % 50 == 0: 
                print(f"\rEpoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # --- 驗證 ---
        print(f"\n   正在計算驗證分數...", end="")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(val_loader):
                if i > 500: break # 只測前 8000 張，節省時間
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\r✅ Epoch {epoch+1} 結束 ({(time.time()-start_time)/60:.1f} min) | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        torch.save(model.state_dict(), f"dino_2080_epoch_{epoch+1}.pth")

    # --- 自動繪圖 ---
    print("📊 [7/7] 正在生成報告圖表...")
    
    # 圖 1: 訓練趨勢
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss (DINOv2)')
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1-Score', color='green', marker='o')
    plt.title('Validation F1-Score (DINOv2)')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig('dino_training_curves.png')

    # 圖 2: 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (DINOv2)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('dino_confusion_matrix.png')

    print("🏆 全部完成！請查看 'dino_training_curves.png' 與 'dino_confusion_matrix.png'。")
    print(f"最終 F1-Score: {history['val_f1'][-1]:.4f}")